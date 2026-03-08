import os
import json
import numpy as np
import pickle
import re
from datetime import datetime

# Mocking the constants and models that app.py would have
CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

MODEL_DIR = "models"

def simulate_predict(text):
    print(f"\n--- Simulating for: '{text}' ---")
    
    # Load Models (simulating the global load)
    model = None
    vectorizer = None
    try:
        with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")

    # Logic from app.py
    prediction_label = "non-offensive"
    category = CATEGORIES[0]
    confidence = 0.8
    is_blacklisted = False
    trigger_source = "None"
    
    # 0. Check Overrides
    overrides_file = "overrides.json"
    if os.path.exists(overrides_file):
        with open(overrides_file, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
        lowered_text = text.lower()
        if lowered_text in overrides:
            label = overrides[lowered_text]
            print(f"OVERRIDE MATCH: {label}")
            return
            
    # 0.5. Check Bad Words
    bad_words_file = "bad_words.json"
    if os.path.exists(bad_words_file):
        with open(bad_words_file, 'r', encoding='utf-8') as f:
            bad_words = json.load(f)
        lowered_text = text.lower()
        for word in bad_words:
            if not word: continue
            pattern = r'\b' + re.escape(word) + r'\b' if len(word) <= 3 else re.escape(word)
            if re.search(pattern, lowered_text):
                is_blacklisted = True
                prediction_label = "offensive"
                category = "Abusive (Blacklisted)"
                confidence = 1.0
                trigger_source = f"Blacklist ({word})"
                break

    # 1. Use SVM
    if not is_blacklisted:
        if model and vectorizer:
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            try:
                decision_score = model.decision_function(X)[0]
                confidence = min(abs(decision_score), 1.0)
                if prediction == 1:
                    if confidence > 0.15:
                        prediction_label = 'offensive'
                        category = "General Offensive"
                        trigger_source = "SVM"
                    else:
                        trigger_source = "SVM (Low Confidence - Ignored)"
                else:
                    trigger_source = "SVM (Safe)"
            except Exception as e:
                prediction_label = 'offensive' if int(prediction) == 1 else 'non-offensive'
                trigger_source = "SVM (Error)"

    print(f"Verdict: {prediction_label}")
    print(f"Source: {trigger_source}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    simulate_predict("Hello")
    simulate_predict("Kutta")
    simulate_predict("Nawaz Sharif")
