"""
Personal Word Override Routes
Allows users to whitelist or blacklist specific words for their own account.
"""

import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify

from database.connection import user_overrides_col
from auth.middleware import token_required, get_optional_user

override_bp = Blueprint('override', __name__)

LOCAL_OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_overrides.json')


def _load_local_user_overrides():
    if os.path.exists(LOCAL_OVERRIDES_FILE):
        try:
            with open(LOCAL_OVERRIDES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_local_user_overrides(data):
    try:
        with open(LOCAL_OVERRIDES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving local user overrides: {e}")


def get_personal_override(username, text_lower):
    """Return override doc for user + text, checking Mongo then local JSON."""
    if not username or username == 'anonymous':
        return None
    if user_overrides_col is not None:
        try:
            doc = user_overrides_col.find_one({'username': username, 'text': text_lower})
            if doc:
                return doc
        except Exception:
            pass
    for item in _load_local_user_overrides():
        if item.get('username') == username and item.get('text') == text_lower:
            return item
    return None


@override_bp.route('/api/override/set', methods=['POST'])
@token_required
def set_override(current_user):
    """Set or update a personal word override for the logged-in user."""
    try:
        data = request.get_json()
        if not data or not data.get('text') or not data.get('label'):
            return jsonify({'error': 'text and label are required'}), 400

        text = data['text'].strip().lower()
        label = data['label'].strip().lower()
        if label not in ['offensive', 'non-offensive']:
            return jsonify({'error': 'label must be offensive or non-offensive'}), 400

        username = current_user.get('username')
        now = datetime.utcnow()

        if user_overrides_col is not None:
            try:
                user_overrides_col.update_one(
                    {'username': username, 'text': text},
                    {'$set': {'label': label, 'updated_at': now}},
                    upsert=True
                )
            except Exception as dbe:
                print(f"MongoDB override save failed, falling back to local: {dbe}")

        items = _load_local_user_overrides()
        updated = False
        for item in items:
            if item.get('username') == username and item.get('text') == text:
                item['label'] = label
                item['updated_at'] = now.isoformat()
                updated = True
                break
        if not updated:
            items.append({
                'username': username,
                'text': text,
                'label': label,
                'created_at': now.isoformat(),
                'updated_at': now.isoformat()
            })
        _save_local_user_overrides(items)

        return jsonify({'message': 'Override saved successfully', 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@override_bp.route('/api/override/list', methods=['GET'])
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


@override_bp.route('/api/override/delete', methods=['POST'])
@token_required
def delete_override(current_user):
    """Delete a personal word override."""
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'error': 'text is required'}), 400

        text = data['text'].strip().lower()
        username = current_user.get('username')

        if user_overrides_col is not None:
            try:
                user_overrides_col.delete_one({'username': username, 'text': text})
            except Exception as dbe:
                print(f"MongoDB delete failed: {dbe}")

        items = _load_local_user_overrides()
        items = [i for i in items if not (i.get('username') == username and i.get('text') == text)]
        _save_local_user_overrides(items)

        return jsonify({'message': 'Override deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
