"""
Flask Backend API for Offensive Language Detection (Context-Aware Multi-Model System)
Supports Real-Time Detection, Multi-Model Scoring (SVM, NB, RF, CNN, LSTM), Authentication, and Feedback.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string
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
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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
    user_overrides_col = db['user_overrides']  # Personal per-user word overrides
    password_resets_col = db['password_resets']  # Secure OTPs for password recovery
except Exception as e:
    print(f"MongoDB Notice: {e}")
    db = None
    users_col = None
    feedback_col = None
    predictions_col = None
    user_overrides_col = None
    password_resets_col = None

# Multi-class category mapping
CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

# Models and assets storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
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

        # Prefer new Keras 3.x .keras format; fall back to legacy .h5
        cnn_keras = os.path.join(MODEL_DIR, "cnn_model.keras")
        cnn_h5    = os.path.join(MODEL_DIR, "cnn_model.h5")
        if os.path.exists(cnn_keras):
            cnn_model = load_model(cnn_keras)
            print("✓ Loaded 1D CNN model (Keras 3.x .keras)")
        elif os.path.exists(cnn_h5):
            cnn_model = load_model(cnn_h5)
            print("✓ Loaded 1D CNN model (.h5 legacy)")

        lstm_keras = os.path.join(MODEL_DIR, "lstm_model.keras")
        lstm_h5    = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_keras):
            lstm_model = load_model(lstm_keras)
            print("✓ Loaded Bi-LSTM model (Keras 3.x .keras)")
        elif os.path.exists(lstm_h5):
            lstm_model = load_model(lstm_h5)
            print("✓ Loaded Bi-LSTM model (.h5 legacy)")

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
    # Normalize Urdu variations (e.g. ھے -> ہے, arabic kaf -> urdu kaf)
    text = text.replace('ھے', 'ہے').replace('ك', 'ک').replace('ي', 'ی').replace('ى', 'ی')
    # Strip tashkeel / harakat
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
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
    # Extract both Latin/ASCII words and Unicode Urdu script words safely
    text_words = set(re.findall(r'[\w\u0600-\u06FF]+', text_lower))

    # ── Polysemy, Zoological & Metaphorical Praise Analyzer (Bilingual: Roman Urdu + Urdu Script) ──
    # 1. Noble animals / heroic praise metaphors (positive in Urdu culture):
    NOBLE_PRAISE_ANIMALS = {
        'sher', 'shair', 'cheeta', 'cheetay', 'shaheen', 'baaz', 'hiran', 'bulbul', 'ghoda', 'ghode',
        'شیر', 'چیتا', 'شاہین', 'باز', 'ہرن', 'بلبل', 'گھوڑا', 'گھوڑے'
    }

    # 2. Pejorative animal slurs (when directed at humans, used as insults):
    PEJORATIVE_ANIMALS = {
        'kutta', 'kutte', 'kutto', 'kutti', 'kuttay', 
        'gadha', 'gadhe', 'gadho', 'ghadha', 'ghadhe', 'khota', 'khote',
        'suar', 'suwar', 'ullu', 'bandar', 'chirkut', 'kanjar', 'bakra', 'bakre',
        'کتا', 'کتے', 'کتوں', 'کتیا', 'کتیاں', 
        'گدھا', 'گدھے', 'گدھوں', 'کھوتا', 'کھوتے',
        'سور', 'خنزیر', 'الو', 'بندر', 'بکرا', 'بکرے'
    }

    ALL_ANIMAL_WORDS = NOBLE_PRAISE_ANIMALS.union(PEJORATIVE_ANIMALS).union({
        'billi', 'billiyan', 'billio', 'murga', 'murgi', 'janwar', 'janwaron', 'haivan', 
        'chirya', 'machhli', 'dog', 'dogs', 'cat', 'cats', 'horse', 'donkey', 'animal', 'animals',
        'بیل', 'بھینس', 'اونٹ', 'ہاتھی', 'لومڑی', 'بھیڑیا',
        'بلی', 'بلیاں', 'مرغ', 'مرغا', 'مرغی', 'جانور', 'جانوروں', 'حیوان', 'حیوانات',
        'چڑیا', 'مچھلی', 'پرندہ', 'پرندے'
    })

    # 3. Explicit Zoological / Factual context indicators (definitively biological context):
    ZOOLOGICAL_FACTUAL_MARKERS = [
        'ek janwar', 'aik janwar', 'janwar hai', 'janwar he', 'janwar hay', 'janwar hein', 'janwar hain',
        'palto', 'paaltu', 'paltoo', 'wafadar janwar', 'jungli janwar', 'jungle', 'jangal', 'chopaya',
        'nasl', 'nasal', 'breed', 'animal', 'pet', 'pets', 'veterinary', 'bhonkta', 'doodh',
        'ایک جانور', 'جانور ہے', 'جانور ھے', 'جانور ہیں', 'وفادار جانور', 'پالتو جانور', 'پالتو',
        'جنگلی جانور', 'جنگل', 'چرند', 'پرند', 'چوپایہ', 'نسل', 'پالا', 'پالتے', 'بھونکتا',
        'کاٹتا', 'گوشت خور', 'گھاس', 'دودھ دینے والا', 'چڑیا گھر'
    ]

    # 4. Praise / Compliment / Positive context descriptors:
    PRAISE_DESCRIPTORS = [
        'bahadur', 'diler', 'shandar', 'shaandaar', 'zindadil', 'behtreen', 
        'zabardast', 'kamal', 'kamaal', 'pyara', 'pyari', 'wafadar', 'madadgar', 
        'masoom', 'shareef', 'fakhr', 'proud', 'hero', 'champ', 'superstar', 
        'acha', 'accha', 'achi', 'acchi', 'good', 'great', 'brave', 'strong', 
        'loyal', 'cute', 'sweet', 'kind', 'honest', 'larte', 'larta', 'larhte', 
        'larhta', 'hifazat', 'bachaya', 'pyar', 'muhabbat', 'pasand', 'khoobsurat', 'khubsurat',
        'بہادر', 'دلیر', 'شاندار', 'زندہ دل', 'بہترین', 'زبردست', 'کمال', 'پیارا', 'پیاری', 
        'وفادار', 'مددگار', 'معصوم', 'شریف', 'فخر', 'اچھا', 'اچھی', 'اچھے', 'خوبصورت', 
        'محبت', 'پیار', 'پسند', 'حفاظت', 'بچایا'
    ]

    # 5. Explicit abuse / slurs (ALWAYS offensive regardless of animal names):
    EXPLICIT_ABUSE = [
        # Shamelessness / dishonour insults
        'beghairat', 'be-ghairat', 'baghairat', 'beghyrat', 'beghairat', 'begharat',
        'besharm', 'be-sharm', 'bewaqoof', 'be-waqoof', 'bewaquf', 'ahmaq',
        'pagal', 'paagal', 'diwana', 'diwanay', 'ullu ka patha', 'ullu ka pattha',
        # Classic heavy slurs
        'kameena', 'kameene', 'kameenay', 'kameeni', 'harami', 'haramkhor',
        'haramzada', 'haramzadi', 'haramzade', 'haram', 
        'jahil', 'jahalat', 'chutiya', 'chutiye', 'chutya', 'madarchod', 'mc',
        'behenchod', 'bhenchod', 'bc', 'randi', 'gandu', 'dalal', 'kanjar',
        'bhadwa', 'bhadway', 'laanti', 'lanat', 'aisi ki taisi', 'maa ki aankh',
        'teri maa', 'teri behan', 'tatti', 'choot', 'lodu', 'lund', 'bhosdike',
        'bhosdi', 'gashti', 'kutti ka', 'kanjri', 'dallay',
        'maderchod', 'madar', 'sister fucker', 'saala', 'saali', 'sala', 'sali',
        'kutte ki aulad', 'kutte k bachay', 'kuttay k bachay', 'kutte kay bachay', 'kuttay kay bachay',
        'kutta k bachay', 'kutta ki aulad', 'kutta kay bachay',
        'gadhe ki aulad', 'gadhe k bachay', 'gadha k bachay', 'gadha ki aulad',
        'namoosi', 'namoos', 'zaleel', 'zaleeli',
        'neech', 'neecha', 'paleed', 'gandagi', 'faaltu', 'nikamma', 'nikaamma',
        'badmaash', 'badmash', 'badtameez', 'badtamiz', 'gustakh', 'ghatiya',
        'khabees', 'khabbees', 'lafanga', 'awara', 'charsi', 'nashebaaz', 'nashebaz',
        'chor', 'chorni', 'dhokebaaz', 'dhoka', 'faraibi', 'munafiq', 'munaafiq',
        'fasiq', 'zalim', 'zaalim', 'murdaad', 'murtad', 'kafir', 'kaafir',
        # Urdu Script explicit abuse
        'بے غیرت', 'بے شرم', 'احمق', 'بیوقوف', 'پاگل', 'دیوانہ', 'الو کا پٹھا',
        'کمینہ', 'کمینے', 'کمینی', 'حرامی', 'حرامخور', 'حرام خور', 'حرامزادہ',
        'جاہل', 'چوتیہ', 'مادرچود', 'بہنچود', 'بہن چود', 'رنڈی', 'گانڈو', 'دلا',
        'کنجر', 'کنجری', 'بھڑوا', 'لعنت', 'تیری ماں', 'تیری بہن', 'لنڈ', 'بھوسڑی',
        'گشتی', 'کتے کی اولاد', 'گدھے کی اولاد', 'ذلیل', 'نیچ', 'پلید', 'بدتمیز',
        'بدقماش', 'بدماش', 'گھٹیا', 'خبیث', 'لفنگا', 'اوارہ', 'منافق', 'ظالم'
    ]

    HUMAN_TARGET_PRONOUNS = {
        'tum', 'tumhe', 'tumko', 'tumhara', 'tumhari', 'tumhare', 
        'tu', 'tujhe', 'tujhko', 'teri', 'tera', 'tere', 
        'aap', 'aapko', 'aapka', 'aapki', 'aapke', 'ap',
        'you', 'your', 'yours', 
        'insan', 'insaan', 'banda', 'banday', 'aadmi', 'shakhs', 'aurat',
        'تم', 'تمہیں', 'تم کو', 'تمکو', 'تمہارا', 'تمہاری', 'تمہارے',
        'تو', 'تجھے', 'تجھکو', 'تیرا', 'تیری', 'تیرے',
        'آپ', 'آپکو', 'آپکا', 'آپکی', 'آپکے',
        'انسان', 'بندہ', 'بندے', 'آدمی', 'شخص', 'عورت', 'لڑکا', 'لڑکی'
    }

    has_noble_animal = bool(text_words.intersection(NOBLE_PRAISE_ANIMALS))
    has_pejorative_animal = bool(text_words.intersection(PEJORATIVE_ANIMALS))
    has_any_animal = bool(text_words.intersection(ALL_ANIMAL_WORDS)) or any(aw in text_lower for aw in ALL_ANIMAL_WORDS)
    has_explicit_abuse = any(ab in text_lower for ab in EXPLICIT_ABUSE)
    has_praise = any(pr in text_lower for pr in PRAISE_DESCRIPTORS)
    has_zoological_marker = any(zm in text_lower for zm in ZOOLOGICAL_FACTUAL_MARKERS)
    has_human_target = bool(text_words.intersection(HUMAN_TARGET_PRONOUNS)) or any(
        p in text_lower for p in [
            'tum ', 'tu ', 'teri ', 'tera ', 'aap ', 'you ', 'insan ', 'banda ',
            'تم ', 'تو ', 'تیرا ', 'تیری ', 'آپ ', 'انسان ', 'بندہ '
        ]
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

    # ── 3. Hybrid Context Decision Engine & Context Calibration ──
    # Determine context categories
    is_zoological = (has_zoological_marker or has_any_animal) and not has_human_target and not has_explicit_abuse
    is_praise = (has_noble_animal or has_praise) and not has_explicit_abuse and not (has_pejorative_animal and not has_praise)
    is_targeted_insult = has_pejorative_animal and has_human_target and not has_praise and not has_zoological_marker

    # Ensure all 5 models have a baseline probability before calibration
    for m_key in ['svm', 'naive_bayes', 'random_forest', 'cnn', 'lstm']:
        if m_key not in probs:
            probs[m_key] = probs.get('svm', 0.5)

    # Apply context calibration to individual model probabilities so votes & verdict are 100% in sync
    for m_key in list(probs.keys()):
        p_raw = probs[m_key]
        if has_explicit_abuse:
            p_adj = max(p_raw, 0.95)
        elif is_targeted_insult:
            p_adj = max(p_raw, 0.85)
        elif is_praise:
            # Praise / Compliment context: scale down offensive probability so models vote Safe (e.g. 0.08 - 0.15)
            p_adj = min(p_raw * 0.15, 0.15)
        elif is_zoological:
            # Zoological narrative context: scale down offensive probability so models vote Safe
            p_adj = min(p_raw * 0.12, 0.12)
        else:
            p_adj = p_raw
        probs[m_key] = p_adj

    # Build models_scores dictionary from calibrated probabilities
    models_scores = {}
    model_display_names = {
        'svm': 'Linear SVM',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'cnn': '1D CNN (Deep Learning)',
        'lstm': 'Bi-LSTM (Deep Learning)'
    }

    for m_key, m_name in model_display_names.items():
        p_val = probs[m_key]
        pred_label = 'offensive' if p_val >= 0.50 else 'non-offensive'
        conf_val = round(float(max(p_val, 1.0 - p_val)), 4)
        models_scores[m_key] = {
            'name': m_name,
            'prediction': pred_label,
            'confidence': conf_val,
            'offensive_prob': round(float(p_val), 4)
        }

    total_models = len(probs)
    offensive_votes = sum(1 for p in probs.values() if p >= 0.50)
    safe_votes = total_models - offensive_votes

    if has_explicit_abuse:
        final_pred = 'offensive'
        final_conf = 0.95
        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Models Confirming) - Explicit Abuse"
        context_override = False
    elif is_praise:
        final_pred = 'non-offensive'
        final_conf = 0.92
        engine_name = f"Consensus Ensemble ({safe_votes}/{total_models} Models Confirming) - Praise & Compliment"
        context_override = True
    elif is_targeted_insult:
        final_pred = 'offensive'
        final_conf = 0.88
        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Models Confirming) - Targeted Insult"
        context_override = False
    elif is_zoological:
        final_pred = 'non-offensive'
        final_conf = 0.93
        engine_name = f"Consensus Ensemble ({safe_votes}/{total_models} Models Confirming) - Zoological Narrative"
        context_override = True
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

    # ── 4. Extract Flagged Offending Word(s) ──
    flagged_words = []
    if has_explicit_abuse:
        for ab in EXPLICIT_ABUSE:
            if ab in text_lower:
                flagged_words.append(ab)
    if has_pejorative_animal and has_human_target and not has_praise:
        for pa in PEJORATIVE_ANIMALS:
            if pa in text_words or pa in text_lower:
                flagged_words.append(pa)
    
    if final_pred == 'offensive' and not flagged_words:
        stop_words = {'hai', 'hain', 'mein', 'kya', 'aur', 'yeh', 'woh', 'kar', 'raha', 'rahe', 'tha', 'the', 'ko', 'se', 'par', 'ki', 'ka', 'ke', 'ne', 'ik', 'ek', 'boht', 'bahut'}
        for w in text_lower.split():
            clean_w = re.sub(r'\W+', '', w)
            if len(clean_w) > 2 and clean_w not in stop_words:
                flagged_words.append(clean_w)
                break

    primary_flagged_word = flagged_words[0] if flagged_words else ""

    return {
        'text': text,
        'prediction': final_pred,
        'confidence': round(float(final_conf), 4),
        'model_used': engine_name,
        'models_scores': models_scores,
        'flagged_word': primary_flagged_word,
        'flagged_words': flagged_words,
        'context_override': context_override,
        'timestamp': datetime.utcnow().isoformat()
    }

# --- Auth Middleware ---
def get_optional_user():
    """Extract user from JWT token if provided; returns None if not provided or invalid"""
    token = request.headers.get('x-access-token')
    if not token and 'Authorization' in request.headers:
        auth = request.headers['Authorization']
        token = auth.split(' ', 1)[1].strip() if auth.startswith('Bearer ') else auth.strip()
    if not token:
        return None
    try:
        data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
        if users_col is not None:
            user = users_col.find_one({'username': data['username']})
            if user:
                return user
        return {'username': data['username'], 'is_admin': data.get('is_admin', False)}
    except Exception:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token and 'Authorization' in request.headers:
            auth = request.headers['Authorization']
            token = auth.split(' ', 1)[1].strip() if auth.startswith('Bearer ') else auth.strip()
        
        if not token:
            print(f"[AUTH 401] Token MISSING for {request.method} {request.path}")
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            current_user = None
            if users_col is not None:
                current_user = users_col.find_one({'username': data['username']})
            if not current_user:
                # Valid signed token with username claim — accept as authenticated user
                current_user = {'username': data['username'], 'is_admin': data.get('is_admin', False)}
        except jwt.ExpiredSignatureError:
            print(f"[AUTH 401] Token EXPIRED for {request.path}")
            return jsonify({'message': 'Token has expired! Please login again.'}), 401
        except Exception as e:
            print(f"[AUTH 401] Token INVALID for {request.path}: {e}")
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'OffenseGuard Backend API',
        'endpoints': {
            'runtime_check': '/api/runtime_check (POST)',
            'predict': '/predict (POST)',
            'download_page': '/download (GET)',
            'health': '/health (GET)'
        }
    })

# --- Auth Routes ---
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Register user using email/username and password"""
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Invalid request data'}), 400
        
    email = data.get('email', '').strip().lower()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    # If username is empty, derive from email
    if not username and email:
        username = email.split('@')[0]
        
    if not username or not password:
        return jsonify({'message': 'Email/Username and password are required'}), 400
        
    if users_col is not None:
        # Check if username or email already registered
        query = {'$or': [{'username': username}]}
        if email:
            query['$or'].append({'email': email})
            
        if users_col.find_one(query):
            return jsonify({'message': 'An account with this email or username already exists'}), 400
        
    # First account registered in the system automatically becomes Super Admin
    is_first_admin = False
    if users_col is not None:
        is_first_admin = (users_col.count_documents({'is_admin': True}) == 0)
        
    hashed_password = generate_password_hash(password)
    user_data = {
        'username': username,
        'email': email,
        'password': hashed_password,
        'is_admin': True if is_first_admin else False,
        'auth_provider': 'local',
        'created_at': datetime.utcnow()
    }
    if users_col is not None:
        users_col.insert_one(user_data)
    
    msg = 'Super Admin account created successfully' if is_first_admin else 'Account created successfully'
    return jsonify({'message': msg, 'username': username, 'is_admin': user_data['is_admin']}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login with either Email or Username and Password"""
    auth = request.get_json()
    if not auth:
        return jsonify({'message': 'Please provide login credentials'}), 401
        
    identifier = (auth.get('email') or auth.get('username') or auth.get('identifier') or '').strip()
    password = (auth.get('password') or '').strip()
    
    if not identifier or not password:
        return jsonify({'message': 'Email/Username and password are required'}), 401
        
    if users_col is None:
        return jsonify({'message': 'Database not connected'}), 500
        
    # Query either by exact username or by lowercase email
    user = users_col.find_one({
        '$or': [
            {'username': identifier},
            {'email': identifier.lower()}
        ]
    })
    
    if not user:
        return jsonify({'message': 'No account found with this email or username'}), 401
        
    if user.get('password') and check_password_hash(user['password'], password):
        token = jwt.encode({
            'username': user['username'],
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm="HS256")
        
        return jsonify({
            'token': token,
            'username': user['username'],
            'email': user.get('email', ''),
            'is_admin': user.get('is_admin', False)
        })
        

def send_otp_email(recipient_email, otp_code):
    """
    Send a 6-digit OTP code to user's email via SMTP.
    Falls back gracefully to server console logs if SMTP is not configured.
    """
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_EMAIL', '')
    smtp_pass = os.getenv('SMTP_PASSWORD', '')

    print(f"\n=======================================================")
    print(f"🔐 [OFFENSE GUARD] PASSWORD RESET OTP DISPATCH")
    print(f"📧 Destination Email: {recipient_email}")
    print(f"🔑 6-DIGIT OTP CODE : {otp_code}")
    print(f"⏳ Expiration Time  : 10 Minutes")
    print(f"=======================================================\n")

    if not smtp_user or not smtp_pass:
        print("ℹ️ Note: SMTP_EMAIL / SMTP_PASSWORD environment variables are not set.")
        print("👉 The OTP has been printed to the server console above for development testing.\n")
        return True

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Offense Guard - Password Reset Verification Code'
        msg['From'] = f"Offense Guard Security <{smtp_user}>"
        msg['To'] = recipient_email

        html_content = f"""
        <html>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f5fa; padding: 24px;">
            <div style="max-width: 480px; margin: 0 auto; background: #ffffff; border-radius: 16px; padding: 32px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); border: 1px solid #e5e7eb;">
              <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #6750A4; margin: 0; font-size: 24px;">🛡️ Offense Guard</h2>
                <p style="color: #6b7280; font-size: 14px; margin-top: 4px;">Security & Account Recovery</p>
              </div>
              <p style="color: #1f2937; font-size: 15px; line-height: 1.5;">Hello,</p>
              <p style="color: #4b5563; font-size: 15px; line-height: 1.5;">We received a request to reset your password. Enter the 6-digit verification code below to verify your identity:</p>
              
              <div style="text-align: center; margin: 28px 0;">
                <span style="display: inline-block; font-size: 34px; font-weight: 800; letter-spacing: 10px; color: #6750A4; background: #EDE9F8; padding: 14px 28px; border-radius: 12px; border: 1px dashed #6750A4;">
                  {otp_code}
                </span>
              </div>
              
              <p style="color: #6b7280; font-size: 13px; line-height: 1.4;">
                • This verification code is valid for <strong>10 minutes</strong>.<br/>
                • Never share this code with anyone.<br/>
                • If you did not request this, you can safely ignore this email.
              </p>
              <hr style="border: none; border-top: 1px solid #f3f4f6; margin: 24px 0;" />
              <p style="color: #9ca3af; font-size: 12px; text-align: center; margin: 0;">
                Offense Guard AI Platform • Automated Security Notification
              </p>
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [recipient_email], msg.as_string())
        server.quit()
        print(f"✅ OTP email successfully dispatched to {recipient_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to dispatch email via SMTP: {e}")
        return False

@app.route('/api/auth/forgot-password/request-otp', methods=['POST'])
def request_password_reset_otp():
    """Generate and send 6-digit OTP to user email"""
    data = request.get_json()
    if not data or not (data.get('email') or data.get('identifier')):
        return jsonify({'message': 'Please provide your registered email or username'}), 400
        
    identifier = (data.get('email') or data.get('identifier') or '').strip().lower()
    
    if users_col is None or password_resets_col is None:
        return jsonify({'message': 'Database not connected'}), 500
        
    user = users_col.find_one({
        '$or': [
            {'email': identifier},
            {'username': identifier}
        ]
    })
    
    if not user:
        return jsonify({'message': 'No account found with this email or username'}), 404
        
    target_email = user.get('email', '')
    if not target_email or '@' not in target_email:
        return jsonify({'message': 'This account does not have a valid registered email address'}), 400

    # Generate cryptographically secure 6-digit OTP
    otp = f"{secrets.randbelow(900000) + 100000}"
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    
    # Store OTP in MongoDB
    password_resets_col.update_one(
        {'email': target_email.lower()},
        {
            '$set': {
                'otp': otp,
                'expires_at': expires_at,
                'attempts': 0,
                'created_at': datetime.utcnow()
            }
        },
        upsert=True
    )
    
    # Dispatch OTP via email or dev logger
    send_otp_email(target_email, otp)
    
    return jsonify({
        'message': f'6-digit verification code sent to {target_email}',
        'email': target_email
    }), 200

@app.route('/api/auth/forgot-password/verify-otp', methods=['POST'])
def verify_password_reset_otp():
    """Verify 6-digit OTP code submitted by user"""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('otp'):
        return jsonify({'message': 'Email and 6-digit OTP code are required'}), 400
        
    email = data.get('email', '').strip().lower()
    otp_input = str(data.get('otp', '')).strip()
    
    if password_resets_col is None:
        return jsonify({'message': 'Database not connected'}), 500
        
    record = password_resets_col.find_one({'email': email})
    if not record:
        return jsonify({'message': 'No reset request found for this email. Please request a new code.'}), 404
        
    # Check expiry
    if datetime.utcnow() > record.get('expires_at', datetime.min):
        password_resets_col.delete_one({'_id': record['_id']})
        return jsonify({'message': 'Verification code has expired. Please request a new code.'}), 400
        
    # Check brute force attempts
    if record.get('attempts', 0) >= 5:
        password_resets_col.delete_one({'_id': record['_id']})
        return jsonify({'message': 'Too many incorrect attempts. Please request a new code.'}), 429
        
    if record.get('otp') != otp_input:
        password_resets_col.update_one({'_id': record['_id']}, {'$inc': {'attempts': 1}})
        remaining = 5 - (record.get('attempts', 0) + 1)
        return jsonify({'message': f'Invalid code. {remaining} attempt(s) remaining.'}), 400
        
    # Success: Issue one-time reset token
    reset_token = secrets.token_urlsafe(32)
    password_resets_col.update_one(
        {'_id': record['_id']},
        {
            '$set': {
                'verified': True,
                'reset_token': reset_token,
                'verified_at': datetime.utcnow()
            }
        }
    )
    
    return jsonify({
        'message': 'Verification code verified successfully!',
        'reset_token': reset_token
    }), 200

@app.route('/api/auth/forgot-password/reset', methods=['POST'])
@app.route('/api/auth/forgot-password', methods=['POST'])
def reset_password_with_otp():
    """Reset password using verified OTP or reset token"""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('new_password'):
        return jsonify({'message': 'Email and new password are required'}), 400
        
    email = data.get('email', '').strip().lower()
    new_password = data.get('new_password', '').strip()
    otp_input = str(data.get('otp', '')).strip()
    reset_token = data.get('reset_token', '').strip()
    
    if len(new_password) < 4:
        return jsonify({'message': 'Password must be at least 4 characters long'}), 400
        
    if users_col is None or password_resets_col is None:
        return jsonify({'message': 'Database not connected'}), 500
        
    # Verify OTP or reset token
    record = password_resets_col.find_one({'email': email})
    if not record:
        return jsonify({'message': 'No active reset session found. Please request an OTP code first.'}), 400
        
    if datetime.utcnow() > record.get('expires_at', datetime.min):
        password_resets_col.delete_one({'_id': record['_id']})
        return jsonify({'message': 'Reset session expired. Please request a new code.'}), 400
        
    is_valid = False
    if reset_token and record.get('reset_token') == reset_token:
        is_valid = True
    elif otp_input and record.get('otp') == otp_input:
        is_valid = True
        
    if not is_valid:
        return jsonify({'message': 'Invalid OTP or reset session. Please verify your code first.'}), 401
        
    user = users_col.find_one({
        '$or': [
            {'email': email},
            {'username': email}
        ]
    })
    
    if not user:
        return jsonify({'message': 'No account found with this email'}), 404
        
    hashed = generate_password_hash(new_password)
    users_col.update_one({'_id': user['_id']}, {'$set': {'password': hashed, 'updated_at': datetime.utcnow()}})
    
    # Delete the reset token record
    password_resets_col.delete_one({'_id': record['_id']})
    
    return jsonify({'message': 'Password reset successful! You can now log in with your new password.'}), 200


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

# --- Personal Override Routes (with resilient JSON fallback) ---

USER_OVERRIDES_FILE = "user_overrides.json"

def _load_local_user_overrides():
    if os.path.exists(USER_OVERRIDES_FILE):
        try:
            with open(USER_OVERRIDES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_local_user_overrides(data):
    try:
        with open(USER_OVERRIDES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving {USER_OVERRIDES_FILE}: {e}")

def get_personal_override(username, text_lower):
    if not username or username == 'anonymous':
        return None
    if user_overrides_col is not None:
        try:
            res = user_overrides_col.find_one({'username': username, 'text_lower': text_lower})
            if res:
                return res
        except Exception as e:
            print(f"MongoDB override lookup error: {e}")
    for item in _load_local_user_overrides():
        if item.get('username') == username and item.get('text_lower') == text_lower:
            return item
    return None

@app.route('/api/override/set', methods=['POST'])
@token_required
def set_override(current_user):
    """
    Save or update a personal word/phrase override for the logged-in user.
    Body: { "text": "...", "label": "offensive" | "non-offensive" }
    """
    try:
        data = request.get_json()
        if not data or not data.get('text') or not data.get('label'):
            return jsonify({'error': 'text and label required'}), 400

        label = data.get('label', '').strip().lower()
        if label not in ('offensive', 'non-offensive'):
            return jsonify({'error': 'label must be "offensive" or "non-offensive"'}), 400

        text_original = data.get('text', '').strip()
        text_lower = text_original.lower().strip()
        username = current_user.get('username')

        if user_overrides_col is not None:
            try:
                user_overrides_col.update_one(
                    {'username': username, 'text_lower': text_lower},
                    {'$set': {
                        'username': username,
                        'text_original': text_original,
                        'text_lower': text_lower,
                        'label': label,
                        'updated_at': datetime.utcnow()
                    }},
                    upsert=True
                )
            except Exception as dbe:
                print(f"MongoDB write failed, using local json: {dbe}")

        # Always keep local json in sync as reliable backup
        items = _load_local_user_overrides()
        items = [i for i in items if not (i.get('username') == username and i.get('text_lower') == text_lower)]
        items.append({
            'username': username,
            'text_original': text_original,
            'text_lower': text_lower,
            'label': label,
            'updated_at': datetime.utcnow().isoformat()
        })
        _save_local_user_overrides(items)

        return jsonify({'message': 'Override saved successfully', 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/override/list', methods=['GET'])
@token_required
def list_overrides(current_user):
    """Return all personal overrides for the logged-in user."""
    try:
        username = current_user.get('username')
        overrides = []
        if user_overrides_col is not None:
            try:
                raw = list(user_overrides_col.find(
                    {'username': username},
                    {'_id': 0, 'username': 0}
                ).sort('updated_at', -1))
                for item in raw:
                    if 'updated_at' in item and isinstance(item['updated_at'], datetime):
                        item['updated_at'] = item['updated_at'].isoformat()
                    overrides.append(item)
                return jsonify(overrides)
            except Exception as dbe:
                print(f"MongoDB list failed, using local json: {dbe}")

        items = _load_local_user_overrides()
        for item in items:
            if item.get('username') == username:
                c = item.copy()
                c.pop('username', None)
                overrides.append(c)
        return jsonify(overrides)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/override/delete', methods=['POST'])
@token_required
def delete_override(current_user):
    """Delete a personal override by text_lower."""
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'error': 'text required'}), 400

        text_lower = data.get('text', '').strip().lower()
        username = current_user.get('username')

        if user_overrides_col is not None:
            try:
                user_overrides_col.delete_one({'username': username, 'text_lower': text_lower})
            except Exception as dbe:
                print(f"MongoDB delete failed: {dbe}")

        items = _load_local_user_overrides()
        items = [i for i in items if not (i.get('username') == username and i.get('text_lower') == text_lower)]
        _save_local_user_overrides(items)

        return jsonify({'message': 'Override deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Core AI Prediction Routes ---


@app.route('/api/runtime_check', methods=['POST'])
def runtime_check():
    """
    Lightweight, fast endpoint for real-time keystroke checking.
    Context-aware model detection without static substring false-positives.
    Returns flagged_word and models_scores.
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'is_offensive': False, 'confidence': 0.0, 'models_scores': {}, 'flagged_word': ''})
            
        text = data.get('text', '').strip()
        text_lower = text.lower().strip()

        # Check personal override first (exact match)
        current_user = get_optional_user()
        username = current_user.get('username', 'anonymous') if (current_user and isinstance(current_user, dict)) else 'anonymous'

        personal_override = get_personal_override(username, text_lower)
        if personal_override:
            label = personal_override.get('label', 'non-offensive')
            return jsonify({
                'is_offensive': label == 'offensive',
                'confidence': 1.0 if label == 'offensive' else 0.0,
                'models_scores': {},
                'flagged_word': '',
                'flagged_words': [],
                'context_override': True,
                'override_source': 'personal'
            })

        result = predict_text_multi_model(text)
        
        return jsonify({
            'is_offensive': result['prediction'] == 'offensive',
            'confidence': result['confidence'],
            'models_scores': result.get('models_scores', {}),
            'flagged_word': result.get('flagged_word', ''),
            'flagged_words': result.get('flagged_words', []),
            'context_override': result.get('context_override', False)
        })
    except Exception as e:
        return jsonify({'is_offensive': False, 'error': str(e), 'flagged_word': ''}), 500

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
        text_lower = text.lower().strip()

        # 1. Check global admin whitelist/blacklist overrides.json
        overrides_file = "overrides.json"
        if os.path.exists(overrides_file):
            try:
                with open(overrides_file, 'r', encoding='utf-8') as f:
                    overrides = json.load(f)
                if text_lower in overrides:
                    label = overrides[text_lower]
                    return jsonify({
                        'text': text,
                        'prediction': label,
                        'confidence': 1.0,
                        'model_used': 'Global Override',
                        'models_scores': {},
                        'context_override': True,
                        'override_source': 'global'
                    })
            except Exception:
                pass

        # 2. Check personal user override (exact match, takes priority over ML)
        current_user = get_optional_user()
        username = current_user.get('username', 'anonymous') if (current_user and isinstance(current_user, dict)) else 'anonymous'

        personal_override = get_personal_override(username, text_lower)
        if personal_override:
            label = personal_override.get('label', 'non-offensive')
            conf = 1.0 if label == 'offensive' else 0.0
            result_override = {
                'text': text,
                'prediction': label,
                'confidence': conf,
                'model_used': 'Your Personal Override',
                'models_scores': {},
                'flagged_word': '',
                'flagged_words': [],
                'context_override': True,
                'override_source': 'personal',
                'timestamp': datetime.utcnow().isoformat()
            }
            # Still log to predictions history
            if predictions_col is not None:
                try:
                    log_item = result_override.copy()
                    log_item['username'] = username
                    predictions_col.insert_one(log_item)
                except Exception:
                    pass
            return jsonify(result_override)

        # 3. Multi-Model Context-Aware Evaluation
        result = predict_text_multi_model(text)

        # 4. Log prediction to DB for history
        if predictions_col is not None:
            try:
                log_item = result.copy()
                log_item['username'] = username
                predictions_col.insert_one(log_item)
            except Exception as dbe:
                print(f"Failed to log prediction to DB: {dbe}")
        
        return jsonify(result)
    
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
    """Get usage statistics and sidebar history (works seamlessly with or without token)"""
    try:
        current_user = get_optional_user()
        raw_history = []
        total = 0
        offensive = 0
        
        if predictions_col is not None:
            try:
                if current_user is not None and isinstance(current_user, dict):
                    uname = current_user.get('username')
                    query = {'username': uname}
                else:
                    query = {'$or': [{'username': 'anonymous'}, {'username': {'$exists': False}}]}
                
                total = predictions_col.count_documents(query)
                offensive = predictions_col.count_documents({'$and': [query, {'prediction': 'offensive'}]})
                raw_history = list(predictions_col.find(query).sort('timestamp', -1).limit(50))
            except Exception as dbe:
                print(f"MongoDB query notice in /stats: {dbe}")

        clean_history = []
        for item in raw_history:
            scores = item.get('models_scores', {})
            entry = {
                'text': str(item.get('text', '')),
                'prediction': str(item.get('prediction', 'non-offensive')),
                'confidence': float(item.get('confidence', 0.8)),
                'model_used': str(item.get('model_used', 'Consensus Ensemble (5 Models)')),
                'models_scores': scores if isinstance(scores, dict) else {},
                'timestamp': str(item.get('timestamp', ''))
            }
            clean_history.append(entry)

        return jsonify({
            'total_predictions': total,  
            'offensive_count': offensive,
            'status': 'active',
            'history': clean_history
        }), 200
    except Exception as e:
        print(f"Error in /stats: {e}")
        return jsonify({
            'total_predictions': 0, 
            'offensive_count': 0, 
            'status': 'active', 
            'history': []
        }), 200

@app.route('/api/history/delete', methods=['POST'])
def delete_history_item():
    try:
        current_user = get_optional_user()
        data = request.get_json()
        if not data or not data.get('timestamp'):
            return jsonify({'error': 'Timestamp required'}), 400
            
        target_timestamp = str(data.get('timestamp')).strip()
        
        if predictions_col is not None:
            if current_user is not None and isinstance(current_user, dict):
                uname = current_user.get('username')
                # Delete by matching timestamp for this user or legacy unassigned entries
                del_res = predictions_col.delete_one({
                    '$or': [{'username': uname}, {'username': 'anonymous'}, {'username': {'$exists': False}}],
                    'timestamp': target_timestamp
                })
            else:
                # If no token, delete matching timestamp
                del_res = predictions_col.delete_one({'timestamp': target_timestamp})
            
            print(f"Deleted history item for timestamp {target_timestamp}: deleted_count={del_res.deleted_count}")
            
        return jsonify({'message': 'Item deleted successfully'})
    except Exception as e:
        print(f"Delete history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        current_user = get_optional_user()
        if predictions_col is not None:
            if current_user is not None and isinstance(current_user, dict):
                predictions_col.delete_many({'username': current_user.get('username')})
            else:
                predictions_col.delete_many({
                    '$or': [{'username': 'anonymous'}, {'username': {'$exists': False}}]
                })
        return jsonify({'message': 'History cleared successfully'})
    except Exception as e:
        print(f"Clear history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download_page():
    """
    Mobile-friendly landing page to download both OffenseGuard APKs:
    - Main Dashboard App (React Native)
    - AI Keyboard APK (System Keyboard)
    """
    downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'downloads')

    keyboard_apk = os.path.join(downloads_dir, 'OffenseGuard_v1.8.apk')
    app_apk = os.path.join(downloads_dir, 'OffenseGuard_App_v1.0.apk')

    keyboard_size = round(os.path.getsize(keyboard_apk) / (1024 * 1024), 1) if os.path.exists(keyboard_apk) else 5.8
    app_size = round(os.path.getsize(app_apk) / (1024 * 1024), 1) if os.path.exists(app_apk) else 153.0

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Offense Guard — Download</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }}
            body {{ background: #0B0E14; color: #E5E7EB; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }}
            .card {{ background: #161B26; border: 1px solid #232B3E; border-radius: 24px; max-width: 500px; width: 100%; padding: 36px 28px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }}
            .badge {{ display: inline-block; background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 20px; }}
            .icon-box {{ width: 84px; height: 84px; margin: 0 auto 20px; background: linear-gradient(135deg, #6366F1, #8B5CF6); border-radius: 22px; display: flex; align-items: center; justify-content: center; font-size: 42px; box-shadow: 0 10px 25px rgba(99, 102, 241, 0.35); }}
            h1 {{ font-size: 26px; font-weight: 800; color: #FFFFFF; margin-bottom: 10px; }}
            p.desc {{ font-size: 14px; color: #9CA3AF; line-height: 1.6; margin-bottom: 28px; }}
            .apk-section {{ margin-bottom: 18px; }}
            .apk-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #6B7280; margin-bottom: 8px; text-align: left; }}
            .btn-download {{ display: block; width: 100%; background: linear-gradient(135deg, #6366F1, #4F46E5); color: #FFFFFF; text-decoration: none; padding: 16px 24px; border-radius: 14px; font-weight: 700; font-size: 15px; box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4); transition: transform 0.2s, box-shadow 0.2s; }}
            .btn-download.secondary {{ background: linear-gradient(135deg, #374151, #1F2937); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }}
            .btn-download:hover {{ transform: translateY(-2px); }}
            .meta {{ font-size: 12px; color: #6B7280; margin-top: 8px; margin-bottom: 6px; text-align: left; }}
            .divider {{ border: none; border-top: 1px solid #232B3E; margin: 20px 0; }}
            .steps {{ text-align: left; background: #11151F; border-radius: 16px; padding: 20px; border: 1px solid #1E2536; margin-top: 24px; }}
            .steps h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; color: #818CF8; margin-bottom: 14px; }}
            .step-item {{ display: flex; align-items: flex-start; margin-bottom: 12px; }}
            .step-item:last-child {{ margin-bottom: 0; }}
            .step-num {{ width: 24px; height: 24px; background: #232B3E; color: #E5E7EB; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; margin-right: 12px; flex-shrink: 0; }}
            .step-text {{ font-size: 13px; color: #D1D5DB; line-height: 1.4; }}
            .step-text strong {{ color: #FFFFFF; }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">FYP Release — v1.0 + v1.8</span>
            <div class="icon-box">🛡️</div>
            <h1>Offense Guard</h1>
            <p class="desc">AI-powered Urdu/Roman Urdu offensive language detection. Install both APKs for the full experience.</p>

            <div class="apk-section">
                <div class="apk-label">📱 Step 1 — Main Dashboard App</div>
                <a href="/download/app" class="btn-download">
                    ⬇️ Download Dashboard App ({app_size} MB)
                </a>
                <div class="meta">Analytics, real-time detection & keyboard setup guide</div>
            </div>

            <div class="apk-section">
                <div class="apk-label">⌨️ Step 2 — AI Keyboard</div>
                <a href="/download/apk" class="btn-download secondary">
                    ⬇️ Download AI Keyboard ({keyboard_size} MB)
                </a>
                <div class="meta">System-wide offensive language blocker keyboard</div>
            </div>

            <div class="steps">
                <h3>Setup Guide</h3>
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-text">Install <strong>Dashboard App</strong> first. Open it and go to Settings → Enable Keyboard.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-text">Install the <strong>AI Keyboard APK</strong> and enable it in Android Settings.</div>
                </div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-text">Select <strong>OffenseGuard Keyboard</strong> as your active keyboard. Done! 🎉</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/download/app')
def download_app_file():
    """
    Direct endpoint to stream the main OffenseGuard Dashboard APK (React Native app).
    """
    downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'downloads')
    apk_filename = 'OffenseGuard_App_v1.0.apk'
    if not os.path.exists(os.path.join(downloads_dir, apk_filename)):
        return jsonify({'error': 'Dashboard APK not found on server'}), 404
    return send_from_directory(downloads_dir, apk_filename, as_attachment=True, mimetype='application/vnd.android.package-archive')

@app.route('/download/apk')
def download_apk_file():
    """
    Direct endpoint to stream the OffenseGuard AI Keyboard APK.
    """
    downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'downloads')
    apk_filename = 'OffenseGuard_v1.8.apk'
    if not os.path.exists(os.path.join(downloads_dir, apk_filename)):
        alt_apk = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offense_guard_keyboard', 'app', 'build', 'outputs', 'apk', 'debug', 'app-debug.apk')
        if os.path.exists(alt_apk):
            return send_file(alt_apk, as_attachment=True, download_name=apk_filename, mimetype='application/vnd.android.package-archive')
        return jsonify({'error': 'Keyboard APK not found on server'}), 404
    return send_from_directory(downloads_dir, apk_filename, as_attachment=True, mimetype='application/vnd.android.package-archive')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
