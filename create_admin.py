from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import os

def create_admin():
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGO_URI)
    db = client['offense_guard']
    users_col = db['users']

    username = input("Enter admin username: ")
    password = input("Enter admin password: ")

    if users_col.find_one({'username': username}):
        print(f"User {username} already exists. Updating to admin...")
        users_col.update_one({'username': username}, {'$set': {'is_admin': True}})
        print("Update successful!")
    else:
        hashed_password = generate_password_hash(password)
        user_data = {
            'username': username,
            'password': hashed_password,
            'is_admin': True,
            'created_at': None # or current time
        }
        users_col.insert_one(user_data)
        print(f"Admin user {username} created successfully!")

if __name__ == "__main__":
    create_admin()
