"""
OffenseGuard — Main Flask Application Entry Point
Modular architecture: initializes Flask, CORS, loads AI models, and registers domain blueprints.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. Models & Detection Engine
from models_layer.model_loader import load_all_models
from detection.engine import predict_text_multi_model
from detection.cleaner import clean_text_for_context

# 2. Database & Auth
from database.connection import (
    users_col,
    feedback_col,
    predictions_col,
    user_overrides_col,
    password_resets_col,
    PRESEEDED_USERS
)
from auth.middleware import get_optional_user, token_required, admin_required, SECRET_KEY

# 3. Route Blueprints
from routes.auth_routes import auth_bp
from routes.detection_routes import detection_bp
from routes.override_routes import override_bp, get_personal_override
from routes.admin_routes import admin_bp
from routes.history_routes import history_bp
from routes.download_routes import download_bp

# Initialize Flask App
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Enable CORS for all origins, headers, and methods
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


@app.before_request
def log_request_info():
    if request.path != '/health' and request.path != '/api/runtime_check':
        print(f"--- Request: {request.method} {request.url} ---")
        if request.is_json:
            print(f"Body: {request.get_json()}")


# Root Endpoint
@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'OffenseGuard Backend API',
        'architecture': 'Modular Blueprint System',
        'endpoints': {
            'runtime_check': '/api/runtime_check (POST)',
            'predict': '/predict (POST)',
            'auth': '/api/auth/* (POST)',
            'override': '/api/override/* (GET/POST)',
            'history': '/api/history/* (POST)',
            'admin': '/api/admin/stats (GET)',
            'download_page': '/download (GET)',
            'health': '/health (GET)'
        }
    })


# Register Domain Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(detection_bp)
app.register_blueprint(override_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(history_bp)
app.register_blueprint(download_bp)

# Load AI & DL Models on startup
load_all_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
