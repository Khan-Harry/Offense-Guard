"""
Evaluate Trained Models (CNN, LSTM) and Generate Performance Reports
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_loader import DataLoader

# Configuration
MODEL_DIR = "models"
RESULTS_DIR = "results"
MAX_LEN = 100
CATEGORIES = {
    0: "Neutral",
    1: "Hate Speech",
    2: "Abusive",
    3: "Offensive"
}

def load_resources():
    """Load tokenizer and models"""
    print("Loading resources...")
    
    # Load Tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    print("✓ Tokenizer loaded")
    
    # Load Models
    cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))
    print("✓ CNN Model loaded")
    
    lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    print("✓ LSTM Model loaded")
    
    return tokenizer, cnn_model, lstm_model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and generate reports"""
    print(f"\n--- Evaluating {model_name} ---")
    
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=list(CATEGORIES.values()))
    print("\nClassification Report:")
    print(report)
    
    # Save Report
    with open(os.path.join(RESULTS_DIR, f"{model_name}_report.txt"), "w") as f:
        f.write(f"{model_name} Evaluation\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(CATEGORIES.values()), 
                yticklabels=list(CATEGORIES.values()))
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    print(f"✓ Saved confusion matrix to results/{model_name}_confusion_matrix.png")
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    loader = DataLoader()
    loader.load_datasets()
    data = loader.preprocess()
    _, _, test_df = loader.split_data()
    
    # 2. Load Resources
    tokenizer, cnn_model, lstm_model = load_resources()
    
    # 3. Prepare Test Data
    print("\nPreparing test data...")
    sequences = tokenizer.texts_to_sequences(test_df['text'])
    X_test = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y_test = test_df['label'].values
    
    # 4. Evaluate CNN
    evaluate_model(cnn_model, X_test, y_test, "CNN")
    
    # 5. Evaluate LSTM
    evaluate_model(lstm_model, X_test, y_test, "LSTM")
    
    print("\n✅ Evaluation Complete. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
