import pickle
import os
import numpy as np
import sys
import json

# Force CPU for tensorflow to avoid GPU issues in test
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_DIR = "models"
CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

def test():
    # 1. Load SVM
    svm_model = None
    vectorizer = None
    try:
        with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
            svm_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        print("✓ SVM loaded")
    except Exception as e:
        print(f"✗ SVM fail: {e}")

    # 2. Load LSTM
    lstm_model = None
    tokenizer = None
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            print("✓ LSTM loaded")
        
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ LSTM fail: {e}")

    test_phrases = ["Hello", "How are you", "Pakistan zindabad", "Kutta", "Nawaz Sharif"]
    
    print("\n" + "="*50)
    print(f"{'Phrase':<20} | {'SVM Pred':<10} | {'LSTM Pred':<10}")
    print("-" * 50)
    
    for text in test_phrases:
        svm_res = "N/A"
        if svm_model and vectorizer:
            X_svm = vectorizer.transform([text])
            svm_pred = svm_model.predict(X_svm)[0]
            svm_res = "OFF" if svm_pred == 1 else "SAFE"
            
        lstm_res = "N/A"
        if lstm_model and tokenizer:
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            preds = lstm_model.predict(padded, verbose=0)[0]
            class_idx = np.argmax(preds)
            lstm_res = "OFF" if class_idx > 0 else "SAFE"
            
        print(f"{text:<20} | {svm_res:<10} | {lstm_res:<10}")

if __name__ == "__main__":
    test()
