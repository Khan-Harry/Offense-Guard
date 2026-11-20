"""
Authentication & Authorization Middleware
Handles token verification for both Firebase Auth JWT and legacy Flask JWT.
Provides @token_required and @admin_required decorators.
"""

import os
import jwt
from functools import wraps
from datetime import datetime
from flask import request, jsonify
from database.connection import users_col

SECRET_KEY = os.getenv('SECRET_KEY', 'super_secret_key_for_fyp_project')


def get_optional_user():
    """Extract user from JWT/Firebase token if provided; returns None if not provided or invalid."""
    token = request.headers.get('x-access-token')
    if not token and 'Authorization' in request.headers:
        auth = request.headers['Authorization']
        token = auth.split(' ', 1)[1].strip() if auth.startswith('Bearer ') else auth.strip()
    if not token:
        return None
    try:
        # Check if Firebase Token
        unverified = jwt.decode(token, options={"verify_signature": False})
        if unverified.get('aud') == 'offenseguard-f8fb2' or 'securetoken.google.com' in unverified.get('iss', ''):
            if unverified.get('exp') and datetime.utcnow().timestamp() > unverified['exp']:
                return None
            email = unverified.get('email', '')
            username = email.split('@')[0] if email else unverified.get('user_id', 'user')
            is_admin = (
                email.lower().startswith('admin') or
                'admin' in email.lower() or
                email.lower() == 'khan@offenseguard.com' or
                unverified.get('admin', False) or
                unverified.get('is_admin', False)
            )
            return {'username': username, 'email': email, 'is_admin': is_admin}

        # Legacy Flask JWT
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
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
            # 1. Try Firebase Token
            unverified = jwt.decode(token, options={"verify_signature": False})
            if unverified.get('aud') == 'offenseguard-f8fb2' or 'securetoken.google.com' in unverified.get('iss', ''):
                if unverified.get('exp') and datetime.utcnow().timestamp() > unverified['exp']:
                    print(f"[AUTH 401] Firebase token EXPIRED for {request.path}")
                    return jsonify({'message': 'Session expired! Please login again.'}), 401

                email = unverified.get('email', '')
                username = email.split('@')[0] if email else unverified.get('user_id', 'user')
                is_admin = (
                    email.lower().startswith('admin') or
                    'admin' in email.lower() or
                    email.lower() == 'khan@offenseguard.com' or
                    unverified.get('admin', False) or
                    unverified.get('is_admin', False)
                )
                current_user = {
                    'username': username,
                    'email': email,
                    'user_id': unverified.get('user_id', unverified.get('sub', '')),
                    'is_admin': is_admin
                }
                return f(current_user, *args, **kwargs)
        except Exception:
            pass

        # 2. Fallback: Legacy Flask JWT
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = None
            if users_col is not None:
                current_user = users_col.find_one({'username': data['username']})
            if not current_user:
                current_user = {'username': data['username'], 'is_admin': data.get('is_admin', False)}
        except jwt.ExpiredSignatureError:
            print(f"[AUTH 401] Token EXPIRED for {request.path}")
            return jsonify({'message': 'Token has expired! Please login again.'}), 401
        except Exception as e:
            print(f"[AUTH 401] Token INVALID for {request.path}: {e}")
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if not current_user.get('is_admin', False):
            return jsonify({'message': 'Admin permission required!'}), 403
        return f(current_user, *args, **kwargs)
    return decorated
