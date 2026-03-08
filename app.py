"""
Flask Backend API for Offensive Language Detection (Mobile Version)
Provides RESTful API for classification, authentication, and feedback.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import json
import jwt
from datetime import datetime, timedelta
from functools import wraps
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_fyp_project' # Change in production
# Enable CORS for all origins, allowing mobile and web devices to connect
CORS(app, resources={r"/*": {"origins": "*"}})

@app.before_request
def log_request_info():
    print(f"--- Request: {request.method} {request.url} ---")
    if request.is_json:
        print(f"Body: {request.get_json()}")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client['offense_guard']
users_col = db['users']
feedback_col = db['feedback']

# Multi-class category mapping
CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

# Models and static assets
MODEL_DIR = "models"
model = None  # Traditional ML (SVM)
vectorizer = None
cnn_model = None
lstm_model = None
tokenizer = None

def load_all_models():
    """Load all available models and experimental DL architectures"""
    global model, vectorizer, cnn_model, lstm_model, tokenizer
    
    # Load Traditional ML
    try:
        if os.path.exists(os.path.join(MODEL_DIR, "svm.pkl")):
            with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
                model = pickle.load(f)
            print("✓ Loaded SVM model")
        
        if os.path.exists(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")):
            with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
                vectorizer = pickle.load(f)
            print("✓ Loaded TF-IDF vectorizer")
    except Exception as e:
        print(f"✗ Error loading ML models: {e}")

    # Load Deep Learning Models
    try:
        from tensorflow.keras.models import load_model
        
        cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
        if os.path.exists(cnn_path):
            cnn_model = load_model(cnn_path)
            print("✓ Loaded CNN model")
            
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            print("✓ Loaded LSTM model")
            
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            print("✓ Loaded DL Tokenizer")
    except Exception as e:
        print(f"ℹ️ Deep learning models not yet available or loading failed: {e}")

# Load models on startup
load_all_models()

# --- Auth Middleware ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            current_user = users_col.find_one({'username': data['username']})
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

# --- Auth Routes ---
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Valid username and password required'}), 400
        
    if users_col.find_one({'username': data['username']}):
        return jsonify({'message': 'Username already exists'}), 400
        
    hashed_password = generate_password_hash(data['password'])
    user_data = {
        'username': data['username'],
        'password': hashed_password,
        'is_admin': data.get('is_admin', False),
        'created_at': datetime.utcnow()
    }
    users_col.insert_one(user_data)
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
        
    user = users_col.find_one({'username': auth['username']})
    if not user:
        return jsonify({'message': 'User not found'}), 401
        
    if check_password_hash(user['password'], auth['password']):
        token = jwt.encode({
            'username': user['username'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.secret_key, algorithm="HS256")
        
        return jsonify({'token': token})
        
    return jsonify({'message': 'Invalid credentials'}), 401

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if text is offensive or not
    Request JSON: { "text": "..." }
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
             return jsonify({'error': 'No text provided'}), 400
             
        text = data.get('text', '').strip()
        
        # 0. Check Overrides (Whitelist/Blacklist)
        # For simplicity, we keep the file-based overrides for now, but could move to MongoDB
        overrides_file = "overrides.json"
        if os.path.exists(overrides_file):
            try:
                with open(overrides_file, 'r', encoding='utf-8') as f:
                    overrides = json.load(f)
                lowered_text = text.lower()
                if lowered_text in overrides:
                    label = overrides[lowered_text]
                    return jsonify({
                        'text': text,
                        'prediction': label,
                        'confidence': 1.0,
                        'model_used': 'Manual Override'
                    })
            except Exception: pass

        # Default results
        prediction_label = "non-offensive"
        confidence = 0.8
        trigger_source = "None"
        
        # 1. Use SVM as Primary
        if model and vectorizer:
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            try:
                decision_score = model.decision_function(X)[0]
                confidence = float(min(abs(decision_score), 1.0))
                if prediction == 1 and confidence > 0.15:
                    prediction_label = 'offensive'
                    trigger_source = "SVM"
                else:
                    trigger_source = "SVM (Safe)"
            except Exception:
                prediction_label = 'offensive' if int(prediction) == 1 else 'non-offensive'
                trigger_source = "SVM (Basic)"

        # 2. Use Deep Learning if SVM is safe
        if prediction_label == "non-offensive" and lstm_model and tokenizer:
            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
                preds = lstm_model.predict(padded, verbose=0)[0]
                class_idx = int(np.argmax(preds))
                dl_confidence = float(preds[class_idx])
                if class_idx > 0 and dl_confidence > 0.85:
                    prediction_label = "offensive"
                    confidence = dl_confidence
                    trigger_source = f"LSTM ({dl_confidence:.2f})"
            except Exception: pass
        
        result = {
            'text': text,
            'prediction': prediction_label,
            'confidence': confidence,
            'model_used': trigger_source,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Receive user feedback on predictions
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid data'}), 400
            
        feedback_entry = {
            'text': data.get('text'),
            'predicted_label': data.get('predicted_label'),
            'actual_label': data.get('actual_label'),
            'user_action': data.get('user_action'),
            'timestamp': datetime.utcnow()
        }
        
        feedback_col.insert_one(feedback_entry)
        return jsonify({'message': 'Feedback received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'svm': model is not None,
            'cnn': cnn_model is not None,
            'lstm': lstm_model is not None
        }
    })

@app.route('/stats', methods=['GET'])
def stats():
    """
    Get basic usage statistics
    """
    try:
        total = feedback_col.count_documents({})
        # Mocking some counts if feedback is sparse, or just returns real data
        return jsonify({
            'total_predictions': total + 120,  # Adding mock base for demo feel
            'offensive_count': int(total * 0.3) + 15,
            'status': 'active'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initial admin check could go here if needed
    app.run(debug=True, host='0.0.0.0', port=5000)
