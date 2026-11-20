"""
Machine Learning and Deep Learning Model Loader
Loads pre-trained models from disk (TF-IDF vectorizer, SVM, Naive Bayes, Random Forest, CNN, LSTM).
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

svm_model = None
nb_model = None
rf_model = None
vectorizer = None

cnn_model = None
lstm_model = None
tokenizer = None


def load_all_models():
    """Load all ML and DL models from disk into global variables."""
    global svm_model, nb_model, rf_model, vectorizer, cnn_model, lstm_model, tokenizer

    # 1. TF-IDF Vectorizer
    try:
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            with open(vec_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("✓ Loaded TF-IDF vectorizer")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")

    # 2. Support Vector Machine (Linear SVM)
    try:
        svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                svm_model = pickle.load(f)
            print("✓ Loaded SVM model (Calibrated)")
    except Exception as e:
        print(f"Error loading SVM: {e}")

    # 3. Naive Bayes (Multinomial NB)
    try:
        nb_path = os.path.join(MODEL_DIR, "nb_model.pkl")
        if os.path.exists(nb_path):
            with open(nb_path, 'rb') as f:
                nb_model = pickle.load(f)
            print("✓ Loaded Naive Bayes model")
    except Exception as e:
        print(f"Error loading Naive Bayes: {e}")

    # 4. Random Forest
    try:
        rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            print("✓ Loaded Random Forest model")
    except Exception as e:
        print(f"Error loading Random Forest: {e}")

    # 5. Deep Learning Tokenizer & Models (CNN, Bi-LSTM)
    try:
        tok_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("✓ Loaded DL Tokenizer")

        from tensorflow.keras.models import load_model

        cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
        if os.path.exists(cnn_path):
            cnn_model = load_model(cnn_path)
            print("✓ Loaded 1D CNN deep learning model")

        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            print("✓ Loaded Bi-LSTM deep learning model")
    except Exception as e:
        print(f"ℹ️ Deep learning loading notice: {e}")
