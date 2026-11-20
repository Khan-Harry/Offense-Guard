"""
Admin & System Health Routes
Provides administrative statistics, public stats, and health check endpoints.
"""

from flask import Blueprint, jsonify
from database.connection import users_col, predictions_col, feedback_col
from auth.middleware import admin_required, get_optional_user
import models_layer.model_loader as ml

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats(current_user):
    """Return aggregated platform metrics for the Admin Dashboard."""
    try:
        total_users = users_col.count_documents({}) if users_col is not None else 0
        total_predictions = predictions_col.count_documents({}) if predictions_col is not None else 0
        offensive_predictions = predictions_col.count_documents({'prediction': 'offensive'}) if predictions_col is not None else 0
        safe_predictions = total_predictions - offensive_predictions

        total_feedback = feedback_col.count_documents({}) if feedback_col is not None else 0
        pending_feedback = feedback_col.count_documents({'verified': {'$ne': True}}) if feedback_col is not None else 0

        # Recent predictions stream
        recent_raw = list(predictions_col.find({}).sort('timestamp', -1).limit(10)) if predictions_col is not None else []
        recent_predictions = []
        for r in recent_raw:
            recent_predictions.append({
                'text': r.get('text', ''),
                'prediction': r.get('prediction', 'non-offensive'),
                'confidence': r.get('confidence', 0.8),
                'timestamp': r.get('timestamp', '')
            })

        return jsonify({
            'total_users': total_users,
            'total_predictions': total_predictions,
            'offensive_predictions': offensive_predictions,
            'safe_predictions': safe_predictions,
            'total_feedback': total_feedback,
            'pending_feedback': pending_feedback,
            'recent_predictions': recent_predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'svm': ml.svm_model is not None,
            'naive_bayes': ml.nb_model is not None,
            'random_forest': ml.rf_model is not None,
            'cnn': ml.cnn_model is not None,
            'lstm': ml.lstm_model is not None
        }
    })


@admin_bp.route('/stats', methods=['GET'])
def stats():
    """Get usage statistics and sidebar history (works seamlessly with or without token)."""
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
