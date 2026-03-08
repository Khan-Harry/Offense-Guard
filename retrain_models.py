"""
Retrain Models using User Feedback (Active Learning)
This script retrains the SVM model based on verified user feedback.
TensorFlow/Deep Learning models are optional.
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = "models"
FEEDBACK_FILE = "feedback_data.json"

def retrain():
    """Retrain SVM model using verified feedback data"""
    print("=== Incremental Learning from Feedback ===")
    
    # 1. Load Feedback Data
    if not os.path.exists(FEEDBACK_FILE):
        print("[ERROR] No feedback data found.")
        return False
    
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        feedback_list = json.load(f)
    
    if not feedback_list:
        print("[ERROR] Feedback list is empty.")
        return False
        
    df = pd.DataFrame(feedback_list)
    
    # 2. Filter for Verified Feedback Only
    if 'verified' not in df.columns:
        df['verified'] = False
        
    initial_count = len(df)
    df = df[df['verified'] == True]
    filtered_count = len(df)
    
    print(f"Loaded {initial_count} feedback samples.")
    print(f"Using {filtered_count} VERIFIED samples for training")
    print(f"Skipped {initial_count - filtered_count} unverified samples")
    
    if filtered_count == 0:
        print("[ERROR] No verified feedback available for retraining.")
        return False
    
    # 3. Prepare Training Data
    # Map labels: 'offensive' -> 1, 'non-offensive' -> 0
    df['label'] = df['actual_label'].apply(lambda x: 1 if x == 'offensive' else 0)
    
    X_feedback = df['text'].values
    y_feedback = df['label'].values
    
    print(f"\nTraining Data Distribution:")
    print(f"   - Offensive: {sum(y_feedback == 1)}")
    print(f"   - Non-offensive: {sum(y_feedback == 0)}")
    
    # Check if we have at least 2 classes
    unique_classes = len(set(y_feedback))
    if unique_classes < 2:
        print(f"\n[ERROR] Need samples from at least 2 classes for training.")
        print(f"[ERROR] Currently have only {unique_classes} class(es).")
        print(f"[INFO] Please approve feedback from both offensive and non-offensive categories.")
        return False
    
    # 4. Load Existing Models
    try:
        with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
            svm_model = pickle.load(f)
        print("[OK] Loaded existing SVM model")
    except Exception as e:
        print(f"[ERROR] Failed to load SVM model: {e}")
        return False
    
    try:
        with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        print("[OK] Loaded TF-IDF vectorizer")
    except Exception as e:
        print(f"[ERROR] Failed to load vectorizer: {e}")
        return False
    
    # 4. Filter and prepare feedback data
    # Only use verified feedback that hasn't been retrained yet
    new_feedback_df = df[(df['verified'] == True) & (df['retrained'] == False)].copy()
    
    if len(new_feedback_df) == 0:
        print("[INFO] No new verified feedback to train on.")
        return False

    print(f"[INFO] Training on {len(new_feedback_df)} new feedback items...")
    
    # 5. Combine with original data for balance
    # Loading a small sample of original data to maintain baseline performance
    ORIGINAL_DATA = "processed_data.csv"
    
    # Prepare feedback data for combination
    feedback_texts = list(new_feedback_df['text'])
    feedback_labels = list(new_feedback_df['actual_label'].map({'offensive': 1.0, 'non-offensive': 0.0}))

    training_texts = feedback_texts
    training_labels = feedback_labels
    
    if os.path.exists(ORIGINAL_DATA):
        try:
            print("[INFO] Loading baseline data for balanced training...")
            baseline_df = pd.read_csv(ORIGINAL_DATA)
            # Take a balanced sample (e.g., 500 safe, 500 offensive)
            safe_sample = baseline_df[baseline_df['label'] == 0.0].sample(n=min(500, len(baseline_df[baseline_df['label'] == 0.0])), random_state=42)
            off_sample = baseline_df[baseline_df['label'] == 1.0].sample(n=min(500, len(baseline_df[baseline_df['label'] == 1.0])), random_state=42)
            
            # Combine feedback with baseline
            training_texts.extend(list(safe_sample['text']))
            training_texts.extend(list(off_sample['text']))
            training_labels.extend(list(safe_sample['label']))
            training_labels.extend(list(off_sample['label']))
            
            print(f"[INFO] Combined training set size: {len(training_labels)}")
            
        except Exception as e:
            print(f"[WARNING] Could not load baseline data: {e}. Training on feedback only.")
    else:
        print(f"[INFO] Original data '{ORIGINAL_DATA}' not found. Training on feedback only.")
    
    # Check if we have at least 2 classes in the combined data
    unique_classes = len(set(training_labels))
    if unique_classes < 2:
        print(f"\n[ERROR] Need samples from at least 2 classes for training.")
        print(f"[ERROR] Currently have only {unique_classes} class(es).")
        print(f"[INFO] Please approve feedback from both offensive and non-offensive categories, or ensure baseline data is balanced.")
        return False

    # Transform combined data
    X_combined = vectorizer.transform(training_texts)
    y_combined = np.array(training_labels)

    # Retrain SVM
    print("\nFine-tuning SVM model...")
    try:
        svm_model.fit(X_combined, y_combined)
        print("[OK] SVM model updated successfully")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return False
    
    # 7. Evaluate on feedback data (only the new feedback items)
    X_feedback_tfidf = vectorizer.transform(new_feedback_df['text'])
    y_feedback = new_feedback_df['actual_label'].map({'offensive': 1.0, 'non-offensive': 0.0})
    
    y_pred = svm_model.predict(X_feedback_tfidf)
    accuracy = accuracy_score(y_feedback, y_pred)
    print(f"\nAccuracy on NEW feedback data: {accuracy:.2%}")
    
    # 8. Save Updated Model
    try:
        with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
            pickle.dump(svm_model, f)
        print("[OK] SVM model saved successfully")
        print(f"[OK] Model saved to: {os.path.join(MODEL_DIR, 'svm.pkl')}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        return False
    
    # 9. Mark feedback as retrained in the JSON file
    try:
        # Create a set of timestamps for verified feedback that was used
        used_timestamps = set(df['timestamp'].values)
        
        # Load all feedback again to preserve non-verified/previously retrained items
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            full_list = json.load(f)
            
        for item in full_list:
            if item.get('timestamp') in used_timestamps:
                item['retrained'] = True
                
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(full_list, f, indent=2)
            
        print("[OK] Feedback data updated with retrained status")
    except Exception as e:
        print(f"[ERROR] Failed to update feedback markers: {e}")
    
    print("\n=== Retraining completed successfully! ===")
    return True

if __name__ == "__main__":
    success = retrain()
    exit(0 if success else 1)
