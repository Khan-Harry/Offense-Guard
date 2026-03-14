from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
import json

client = MongoClient('mongodb://localhost:27017/')
db = client['offense_guard']
users_col = db['users']

print("--- USERS IN DB ---")
for user in users_col.find({}):
    # print username and hash (safely)
    print(f"User: '{user.get('username')}'")
    print(f"Hash: {user.get('password')}")
    
    # Try testing some common passwords like '123456', 'password', or the username itself
    test_pw = user.get('username')
    is_valid = check_password_hash(user.get('password'), test_pw)
    print(f"Password == Username? {is_valid}")

print("--- TEST HASHING ---")
pw = "testing123"
h = generate_password_hash(pw)
print(f"Generated: {h}")
print(f"Verification: {check_password_hash(h, pw)}")
