"""
Flask Backend API for Offensive Language Detection (Context-Aware Multi-Model System)
Supports Real-Time Detection, Multi-Model Scoring (SVM, NB, RF, CNN, LSTM), Authentication, and Feedback.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import json
import jwt
import re
from datetime import datetime, timedelta
from functools import wraps
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'super_secret_key_for_fyp_project')
# Enable CORS for all origins, headers, and methods
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.before_request
def log_request_info():
    if request.path != '/health' and request.path != '/api/runtime_check':
        print(f"--- Request: {request.method} {request.url} ---")
        if request.is_json:
            print(f"Body: {request.get_json()}")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
    db = client['offense_guard']
    users_col = db['users']
    feedback_col = db['feedback']
    predictions_col = db['predictions']
except Exception as e:
    print(f"MongoDB Notice: {e}")
    db = None
    users_col = None
    feedback_col = None
    predictions_col = None

# Multi-class category mapping
CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

# Models and assets storage
MODEL_DIR = "models"
svm_model = None
nb_model = None
rf_model = None
vectorizer = None

cnn_model = None
lstm_model = None
tokenizer = None

def load_all_models():
    """Load all ML and DL models from disk"""
    global svm_model, nb_model, rf_model, vectorizer, cnn_model, lstm_model, tokenizer
    
    # 1. TF-IDF Vectorizer
    try:
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            print("✓ Loaded TF-IDF vectorizer")
    except Exception as e:
        print(f"✗ Error loading TF-IDF vectorizer: {e}")

    # 2. Machine Learning Classifiers
    try:
        svm_path = os.path.join(MODEL_DIR, "svm.pkl")
        if os.path.exists(svm_path):
            with open(svm_path, "rb") as f:
                svm_model = pickle.load(f)
            print("✓ Loaded SVM model (Calibrated)")
        
        nb_path = os.path.join(MODEL_DIR, "naive_bayes.pkl")
        if os.path.exists(nb_path):
            with open(nb_path, "rb") as f:
                nb_model = pickle.load(f)
            print("✓ Loaded Naive Bayes model")
            
        rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, "rb") as f:
                rf_model = pickle.load(f)
            print("✓ Loaded Random Forest model")
    except Exception as e:
        print(f"✗ Error loading ML models: {e}")

    # 3. Deep Learning Models & Tokenizer
    try:
        from tensorflow.keras.models import load_model
        
        tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        if os.path.exists(tok_path):
            with open(tok_path, "rb") as f:
                tokenizer = pickle.load(f)
            print("✓ Loaded DL Tokenizer")
            
        cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
        if os.path.exists(cnn_path):
            cnn_model = load_model(cnn_path)
            print("✓ Loaded 1D CNN model")
            
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            print("✓ Loaded Bi-LSTM model")
    except Exception as e:
        print(f"ℹ️ Deep learning loading notice: {e}")

# Load models on server boot
load_all_models()

def clean_text_for_context(text):
    if not text:
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Normalize common Roman Urdu animal/slur spellings
    text = re.sub(r'\bghadha\b', 'gadha', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkhota\b', 'gadha', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkuttay\b', 'kutta', text, flags=re.IGNORECASE)
    text = re.sub(r'\bkameenay\b', 'kameena', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text_multi_model(text):
    """
    Context-aware multi-model evaluation utilizing all 5 trained models:
    Linear SVM, Multinomial Naive Bayes, Random Forest, 1D CNN, and Bi-LSTM.
    Integrates Polysemy & Zoometaphor Context Analysis to eliminate false positives
    on zoological/domestic animal sentences while detecting true human insults.
    """
    if not text or not text.strip():
        return {
            'prediction': 'non-offensive',
            'confidence': 1.0,
            'model_used': 'None',
            'models_scores': {},
            'context_override': False
        }
        
    cleaned = clean_text_for_context(text)
    text_lower = cleaned.lower()
    text_words = set(re.findall(r'\b\w+\b', text_lower))

    # ── Polysemy, Zoological & Metaphorical Praise Analyzer ──
    # 1. Noble animals / heroic praise metaphors (positive in Urdu culture):
    NOBLE_PRAISE_ANIMALS = {
        'sher', 'shair', 'cheeta', 'cheetay', 'shaheen', 'baaz', 'hiran', 'bulbul', 'ghoda', 'ghode'
    }

    # 2. Pejorative animal slurs (when directed at humans, used as insults):
    PEJORATIVE_ANIMALS = {
        'kutta', 'kutte', 'kutto', 'kutti', 'kuttay', 
        'gadha', 'gadhe', 'gadho', 'ghadha', 'ghadhe', 'khota', 'khote',
        'suar', 'suwar', 'ullu', 'bandar', 'chirkut', 'kanjar', 'bakra', 'bakre'
    }

    ALL_ANIMAL_WORDS = NOBLE_PRAISE_ANIMALS.union(PEJORATIVE_ANIMALS).union({
        'billi', 'billiyan', 'billio', 'murga', 'murgi', 'janwar', 'janwaron', 'haivan', 
        'chirya', 'machhli', 'dog', 'dogs', 'cat', 'cats', 'horse', 'donkey', 'animal', 'animals'
    })

    # 3. Praise / Compliment / Positive context descriptors:
    PRAISE_DESCRIPTORS = [
        'bahadur', 'diler', 'shandar', 'shaandaar', 'zindadil', 'behtreen', 
        'zabardast', 'kamal', 'kamaal', 'pyara', 'pyari', 'wafadar', 'madadgar', 
        'masoom', 'shareef', 'fakhr', 'proud', 'hero', 'champ', 'superstar', 
        'acha', 'accha', 'achi', 'acchi', 'good', 'great', 'brave', 'strong', 
        'loyal', 'cute', 'sweet', 'kind', 'honest', 'larte', 'larta', 'larhte', 
        'larhta', 'hifazat', 'bachaya', 'pyar', 'muhabbat', 'pasand', 'khoobsurat', 'khubsurat'
    ]

    # 4. Explicit abuse / slurs (ALWAYS offensive):
    EXPLICIT_ABUSE = [
        'kameena', 'kameene', 'kameenay', 'kameeni', 'harami', 'haramkhor', 
        'jahil', 'jahalat', 'chutiya', 'chutiye', 'chutya', 'madarchod', 'mc', 
        'behenchod', 'bhenchod', 'bc', 'randi', 'gandu', 'dalal', 'kanjar', 
        'bhadwa', 'bhadway', 'laanti', 'lanat', 'aisi ki taisi', 'maa ki aankh', 
        'teri maa', 'teri behan', 'tatti', 'choot', 'lodu', 'lund', 'bhosdike', 
        'bhosdi', 'gashti', 'kutti ka', 'kanjri', 'dallay'
    ]

    HUMAN_TARGET_PRONOUNS = {
        'tum', 'tumhe', 'tumko', 'tumhara', 'tumhari', 'tumhare', 
        'tu', 'tujhe', 'tujhko', 'teri', 'tera', 'tere', 
        'aap', 'aapko', 'aapka', 'aapki', 'aapke', 'ap',
        'you', 'your', 'yours', 
        'insan', 'insaan', 'banda', 'banday', 'aadmi', 'shakhs', 'aurat'
    }

    has_noble_animal = bool(text_words.intersection(NOBLE_PRAISE_ANIMALS))
    has_pejorative_animal = bool(text_words.intersection(PEJORATIVE_ANIMALS))
    has_any_animal = bool(text_words.intersection(ALL_ANIMAL_WORDS))
    has_explicit_abuse = any(ab in text_lower for ab in EXPLICIT_ABUSE)
    has_praise = any(pr in text_lower for pr in PRAISE_DESCRIPTORS)
    has_human_target = bool(text_words.intersection(HUMAN_TARGET_PRONOUNS)) or any(
        p in text_lower for p in ['tum ', 'tu ', 'teri ', 'tera ', 'aap ', 'you ', 'insan ', 'banda ']
    )

    models_scores = {}
    probs = {}
    
    # ── 1. Machine Learning Models (TF-IDF) ──
    if vectorizer is not None:
        try:
            X = vectorizer.transform([cleaned])
            
            # SVM
            if svm_model is not None:
                if hasattr(svm_model, 'predict_proba'):
                    p_svm = float(svm_model.predict_proba(X)[0][1])
                else:
                    dec = float(svm_model.decision_function(X)[0])
                    p_svm = float(1.0 / (1.0 + np.exp(-dec)))
                probs['svm'] = p_svm
                models_scores['svm'] = {
                    'name': 'Linear SVM',
                    'prediction': 'offensive' if p_svm >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_svm, 1 - p_svm)), 4),
                    'offensive_prob': round(p_svm, 4)
                }
            
            # Naive Bayes
            if nb_model is not None:
                p_nb = float(nb_model.predict_proba(X)[0][1])
                probs['naive_bayes'] = p_nb
                models_scores['naive_bayes'] = {
                    'name': 'Naive Bayes',
                    'prediction': 'offensive' if p_nb >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_nb, 1 - p_nb)), 4),
                    'offensive_prob': round(p_nb, 4)
                }
                
            # Random Forest
            if rf_model is not None:
                p_rf = float(rf_model.predict_proba(X)[0][1])
                probs['random_forest'] = p_rf
                models_scores['random_forest'] = {
                    'name': 'Random Forest',
                    'prediction': 'offensive' if p_rf >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_rf, 1 - p_rf)), 4),
                    'offensive_prob': round(p_rf, 4)
                }
        except Exception as e:
            print(f"ML Prediction Error: {e}")

    # ── 2. Deep Learning Models (CNN, Bi-LSTM) ──
    if tokenizer is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            
            # 1D CNN
            if cnn_model is not None:
                p_cnn = float(cnn_model.predict(padded, verbose=0)[0][1])
                probs['cnn'] = p_cnn
                models_scores['cnn'] = {
                    'name': '1D CNN (Deep Learning)',
                    'prediction': 'offensive' if p_cnn >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_cnn, 1 - p_cnn)), 4),
                    'offensive_prob': round(p_cnn, 4)
                }
                
            # Bi-LSTM
            if lstm_model is not None:
                p_lstm = float(lstm_model.predict(padded, verbose=0)[0][1])
                probs['lstm'] = p_lstm
                models_scores['lstm'] = {
                    'name': 'Bi-LSTM (Deep Learning)',
                    'prediction': 'offensive' if p_lstm >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_lstm, 1 - p_lstm)), 4),
                    'offensive_prob': round(p_lstm, 4)
                }
        except Exception as e:
            print(f"DL Prediction Error: {e}")

    # ── 3. Hybrid Context Decision Engine ──
    total_models = len(probs) if probs else 1
    offensive_votes = sum(1 for p in probs.values() if p >= 0.50)
    
    # Case A: Explicit curses / profanity -> ALWAYS OFFENSIVE
    if has_explicit_abuse:
        final_pred = 'offensive'
        final_conf = 0.95
        engine_name = f"Explicit Abuse / Slur Engine ({offensive_votes}/{total_models} Models Confirming)"
        context_override = False

    # Case B: Praise / Compliments (e.g. "tum sher ki tarah bahadur ho", "tum sher ki tarah larhte ho", "mera sher beta")
    # -> 100% NON-OFFENSIVE (Safe Praise)
    elif (has_noble_animal or has_praise) and not has_explicit_abuse and not (has_pejorative_animal and not has_praise):
        final_pred = 'non-offensive'
        final_conf = 0.94
        engine_name = "Praise & Compliment Engine (Tareefi Jumla Verified)"
        context_override = True

    # Case C: Human target + Pejorative Animal Slur without praise (e.g. "tu gadha hai", "tum aik kutte ho") -> OFFENSIVE
    elif has_pejorative_animal and has_human_target and not has_praise:
        final_pred = 'offensive'
        conf_vals = [p for p in probs.values() if p >= 0.40]
        final_conf = (sum(conf_vals)/len(conf_vals)) if conf_vals else 0.85
        engine_name = f"Metaphorical Human Insult Engine ({offensive_votes}/{total_models} Models Confirming)"
        context_override = False

    # Case D: Animal Word WITHOUT Human Target and WITHOUT Abuse (e.g. "kutta ik acha janwar hai", "meri billi ko kutte ne maar diya")
    # -> 100% NON-OFFENSIVE (Safe Zoological Context)
    elif has_any_animal and not has_human_target and not has_explicit_abuse:
        final_pred = 'non-offensive'
        final_conf = 0.92
        engine_name = "Zoological Context Engine (Safe Animal Narrative Verified)"
        context_override = True

    # Case E: Standard Ensemble Majority Consensus (3/5 Votes)
    else:
        context_override = False
        if offensive_votes >= (total_models / 2.0 + 0.1):
            final_pred = 'offensive'
            agreeing_probs = [p for p in probs.values() if p >= 0.50]
            final_conf = sum(agreeing_probs) / len(agreeing_probs) if agreeing_probs else 0.70
        else:
            final_pred = 'non-offensive'
            safe_probs = [1.0 - p for p in probs.values() if p < 0.50]
            final_conf = sum(safe_probs) / len(safe_probs) if safe_probs else 0.70

        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Votes Offensive: SVM, NB, RF, CNN, LSTM)"

    return {
        'text': text,
        'prediction': final_pred,
        'confidence': round(float(final_conf), 4),
        'model_used': engine_name,
        'models_scores': models_scores,
        'context_override': context_override,
        'timestamp': datetime.utcnow().isoformat()
    }

# --- Auth Middleware ---
def get_optional_user():
    """Extract user from JWT token if provided; returns None if not provided or invalid"""
    token = request.headers.get('x-access-token')
    if not token:
        return None
    try:
        data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
        if users_col is not None:
            return users_col.find_one({'username': data['username']})
        return {'username': data['username']}
    except Exception:
        return None

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
            if users_col is not None:
                current_user = users_col.find_one({'username': data['username']})
            else:
                current_user = {'username': data['username']}
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
        
    if users_col is not None and users_col.find_one({'username': data['username']}):
        return jsonify({'message': 'Username already exists'}), 400
        
    hashed_password = generate_password_hash(data['password'])
    user_data = {
        'username': data['username'],
        'password': hashed_password,
        'is_admin': data.get('is_admin', False),
        'created_at': datetime.utcnow()
    }
    if users_col is not None:
        users_col.insert_one(user_data)
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
        
    if users_col is None:
        return jsonify({'message': 'Database not connected'}), 500
        
    user = users_col.find_one({'username': auth['username']})
    if not user:
        return jsonify({'message': 'User not found'}), 401
        
    if check_password_hash(user['password'], auth['password']):
        token = jwt.encode({
            'username': user['username'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.secret_key, algorithm="HS256")
        
        return jsonify({
            'token': token,
            'username': user['username'],
            'is_admin': user.get('is_admin', False)
        })
        
    return jsonify({'message': 'Invalid credentials'}), 401

# --- Admin Middleware ---
def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if not current_user.get('is_admin', False):
            return jsonify({'message': 'Admin permission required!'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# --- Admin Routes ---
@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats(current_user):
    """Get global system statistics for admin"""
    try:
        total_users = users_col.count_documents({}) if users_col is not None else 0
        total_feedback = feedback_col.count_documents({}) if feedback_col is not None else 0
        pending_feedback = feedback_col.count_documents({'verified': {'$ne': True}}) if feedback_col is not None else 0
        total_predictions = predictions_col.count_documents({}) if predictions_col is not None else 0
        
        recent_users = list(users_col.find({}, {'password': 0}).sort('created_at', -1).limit(10)) if users_col is not None else []
        for user in recent_users:
            user['_id'] = str(user['_id'])
            if 'created_at' in user and isinstance(user['created_at'], datetime):
                user['created_at'] = user['created_at'].isoformat()

        return jsonify({
            'total_users': total_users,
            'total_feedback': total_feedback,
            'pending_feedback': pending_feedback,
            'total_predictions': total_predictions,
            'recent_users': recent_users,
            'models_loaded': {
                'svm': svm_model is not None,
                'naive_bayes': nb_model is not None,
                'random_forest': rf_model is not None,
                'cnn': cnn_model is not None,
                'lstm': lstm_model is not None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/feedback/list', methods=['GET'])
@admin_required
def list_feedback(current_user):
    """List all feedback for verification"""
    try:
        feedback = list(feedback_col.find().sort('timestamp', -1).limit(50)) if feedback_col is not None else []
        for f in feedback:
            f['_id'] = str(f['_id'])
            if 'timestamp' in f and isinstance(f['timestamp'], datetime):
                f['timestamp'] = f['timestamp'].isoformat()
        return jsonify(feedback)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/feedback/verify', methods=['POST'])
@admin_required
def verify_feedback(current_user):
    """Verify or reject user feedback"""
    try:
        data = request.get_json()
        if not data or not data.get('feedback_id'):
            return jsonify({'error': 'Feedback ID required'}), 400
            
        from bson.objectid import ObjectId
        action = data.get('action')
        
        if feedback_col is not None:
            if action == 'verify':
                feedback_col.update_one(
                    {'_id': ObjectId(data['feedback_id'])},
                    {'$set': {'verified': True, 'retrained': False}}
                )
            else:
                feedback_col.delete_one({'_id': ObjectId(data['feedback_id'])})
            
        return jsonify({'message': f'Feedback {action}ed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/retrain', methods=['POST'])
@admin_required
def trigger_retrain(current_user):
    """Manually trigger the retraining process"""
    try:
        import subprocess
        process = subprocess.Popen(['python', 'retrain_models.py'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            load_all_models()
            return jsonify({'message': 'Retraining successful', 'output': stdout})
        else:
            return jsonify({'error': 'Retraining failed', 'details': stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Core AI Prediction Routes ---

@app.route('/api/runtime_check', methods=['POST'])
def runtime_check():
    """
    Lightweight, fast endpoint for real-time keystroke checking.
    Context-aware model detection without static substring false-positives.
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'is_offensive': False, 'confidence': 0.0, 'models_scores': {}})
            
        text = data.get('text', '').strip()
        result = predict_text_multi_model(text)
        
        return jsonify({
            'is_offensive': result['prediction'] == 'offensive',
            'confidence': result['confidence'],
            'models_scores': result.get('models_scores', {}),
            'context_override': result.get('context_override', False)
        })
    except Exception as e:
        return jsonify({'is_offensive': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main Prediction Endpoint:
    Evaluates text context through SVM, Naive Bayes, Random Forest, CNN, and LSTM.
    Returns overall ensemble prediction and individual model breakdown.
    Works seamlessly with or without auth token.
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'error': 'No text provided'}), 400
             
        text = data.get('text', '').strip()
        
        # 1. Check manual whitelist/blacklist overrides if present
        overrides_file = "overrides.json"
        if os.path.exists(overrides_file):
            try:
                with open(overrides_file, 'r', encoding='utf-8') as f:
                    overrides = json.load(f)
                lowered = text.lower()
                if lowered in overrides:
                    label = overrides[lowered]
                    return jsonify({
                        'text': text,
                        'prediction': label,
                        'confidence': 1.0,
                        'model_used': 'Manual Override',
                        'models_scores': {},
                        'context_override': True
                    })
            except Exception:
                pass

        # 2. Multi-Model Context-Aware Evaluation
        result = predict_text_multi_model(text)
        
        # 3. Log prediction to DB for history if user authenticated
        current_user = get_optional_user()
        if predictions_col is not None and current_user is not None and isinstance(current_user, dict):
            try:
                log_item = result.copy()
                log_item['username'] = current_user.get('username', 'anonymous')
                predictions_col.insert_one(log_item)
            except Exception:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Receive user feedback on predictions for active learning"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid data'}), 400
            
        feedback_entry = {
            'text': data.get('text'),
            'predicted_label': data.get('predicted_label'),
            'actual_label': data.get('actual_label'),
            'user_action': data.get('user_action'),
            'timestamp': datetime.utcnow(),
            'verified': False,
            'retrained': False
        }
        
        if feedback_col is not None:
            feedback_col.insert_one(feedback_entry)
        return jsonify({'message': 'Feedback received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'svm': svm_model is not None,
            'naive_bayes': nb_model is not None,
            'random_forest': rf_model is not None,
            'cnn': cnn_model is not None,
            'lstm': lstm_model is not None
        }
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get usage statistics for the current user (optional token)"""
    try:
        current_user = get_optional_user()
        if predictions_col is None or current_user is None or not isinstance(current_user, dict):
            return jsonify({'total_predictions': 0, 'offensive_count': 0, 'status': 'active', 'history': []})
            
        uname = current_user.get('username')
        total = predictions_col.count_documents({'username': uname})
        offensive = predictions_col.count_documents({'username': uname, 'prediction': 'offensive'})
        recent_history = list(predictions_col.find({'username': uname}, {'_id': 0}).sort('timestamp', -1).limit(20))
        
        return jsonify({
            'total_predictions': total,  
            'offensive_count': offensive,
            'status': 'active',
            'history': recent_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/delete', methods=['POST'])
def delete_history_item():
    try:
        current_user = get_optional_user()
        data = request.get_json()
        if not data or not data.get('timestamp'):
            return jsonify({'error': 'Timestamp required'}), 400
            
        if predictions_col is not None and current_user is not None and isinstance(current_user, dict):
            predictions_col.delete_one({
                'username': current_user.get('username'),
                'timestamp': data.get('timestamp')
            })
        return jsonify({'message': 'Item deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        current_user = get_optional_user()
        if predictions_col is not None and current_user is not None and isinstance(current_user, dict):
            predictions_col.delete_many({'username': current_user.get('username')})
        return jsonify({'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
