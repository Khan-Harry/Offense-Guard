"""
Database Connection & Initialization
Connects to MongoDB Atlas if MONGO_URI is set, otherwise falls back gracefully to LocalCollection.
"""

import os
from datetime import datetime
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from database.local_collection import LocalCollection

# Pre-seeded test & demo accounts
PRESEEDED_USERS = [
    {
        '_id': 'admin_001',
        'username': 'admin',
        'email': 'admin@offenseguard.com',
        'password': generate_password_hash('admin123'),
        'is_admin': True,
        'auth_provider': 'local',
        'created_at': datetime.utcnow().isoformat()
    },
    {
        '_id': 'user_001',
        'username': 'user',
        'email': 'user@offenseguard.com',
        'password': generate_password_hash('user123'),
        'is_admin': False,
        'auth_provider': 'local',
        'created_at': datetime.utcnow().isoformat()
    },
    {
        '_id': 'khan_001',
        'username': 'khan',
        'email': 'khan@offenseguard.com',
        'password': generate_password_hash('khan123'),
        'is_admin': True,
        'auth_provider': 'local',
        'created_at': datetime.utcnow().isoformat()
    }
]

MONGO_URI = os.getenv("MONGO_URI")
client = None
db = None
users_col = None
feedback_col = None
predictions_col = None
user_overrides_col = None
password_resets_col = None

if MONGO_URI and not MONGO_URI.startswith("mongodb://localhost"):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        db = client['offense_guard']
        users_col = db['users']
        feedback_col = db['feedback']
        predictions_col = db['predictions']
        user_overrides_col = db['user_overrides']
        password_resets_col = db['password_resets']
        print("✓ Connected to MongoDB Atlas successfully")
    except Exception as e:
        print(f"MongoDB Notice: {e}. Switching to LocalCollection fallback.")
        client = None

if users_col is None:
    users_col = LocalCollection('users', initial_docs=PRESEEDED_USERS)
    feedback_col = LocalCollection('feedback')
    predictions_col = LocalCollection('predictions')
    user_overrides_col = LocalCollection('user_overrides')
    password_resets_col = LocalCollection('password_resets')
    print("✓ LocalCollection fallback online with active pre-seeded accounts")
