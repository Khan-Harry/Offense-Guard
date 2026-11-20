"""
Prediction History Management Routes
Endpoints for deleting single items or clearing history.
"""

from flask import Blueprint, request, jsonify
from database.connection import predictions_col
from auth.middleware import get_optional_user

history_bp = Blueprint('history', __name__)


@history_bp.route('/api/history/delete', methods=['POST'])
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
                del_res = predictions_col.delete_one({
                    '$or': [{'username': uname}, {'username': 'anonymous'}, {'username': {'$exists': False}}],
                    'timestamp': target_timestamp
                })
            else:
                del_res = predictions_col.delete_one({'timestamp': target_timestamp})

            print(f"Deleted history item for timestamp {target_timestamp}: deleted_count={del_res.deleted_count}")

        return jsonify({'message': 'Item deleted successfully'})
    except Exception as e:
        print(f"Delete history error: {e}")
        return jsonify({'error': str(e)}), 500


@history_bp.route('/api/history/clear', methods=['POST'])
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
