import pickle
import os
import sys

MODEL_DIR = "models"
try:
    with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    
    test_phrases = ["Hello", "Good morning", "Urdu language", "Nawaz Sharif", "Kutta"]
    for p in test_phrases:
        X = vectorizer.transform([p])
        pred = model.predict(X)[0]
        print(f"'{p}' -> Prediction: {pred}")
        
except Exception as e:
    print(f"Error: {e}")
