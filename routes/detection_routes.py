"""
Detection Routes
Endpoints for real-time keystroke checking and full multi-model text prediction.
"""

import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify

from database.connection import predictions_col
from auth.middleware import get_optional_user
from routes.override_routes import get_personal_override
from detection.engine import predict_text_multi_model

detection_bp = Blueprint('detection', __name__)

OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'overrides.json')


@detection_bp.route('/api/runtime_check', methods=['POST'])
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

        # 1. Check global admin whitelist/blacklist overrides.json
        if os.path.exists(OVERRIDES_FILE):
            try:
                with open(OVERRIDES_FILE, 'r', encoding='utf-8') as f:
                    overrides = json.load(f)
                if text_lower in overrides:
                    label = overrides[text_lower]
                    return jsonify({
                        'is_offensive': label == 'offensive',
                        'confidence': 1.0 if label == 'offensive' else 0.0,
                        'models_scores': {},
                        'flagged_word': '',
                        'flagged_words': [],
                        'context_override': True,
                        'override_source': 'global'
                    })
            except Exception:
                pass

        # 2. Check personal override first (exact match)
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


@detection_bp.route('/predict', methods=['POST'])
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
        if os.path.exists(OVERRIDES_FILE):
            try:
                with open(OVERRIDES_FILE, 'r', encoding='utf-8') as f:
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
