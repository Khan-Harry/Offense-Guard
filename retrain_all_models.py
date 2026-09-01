"""
Full Retraining Script - All 5 Models on All 11 Datasets (including T1, T2, T3)
Trains: SVM, Naive Bayes, Random Forest, CNN, LSTM
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from data_loader import DataLoader

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load & Preprocess All 11 Datasets
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1: Loading ALL 11 Datasets (incl. T1, T2, T3)")
print("="*60)

loader = DataLoader()
loader.load_datasets()
loader.preprocess()

print(f"\n✓ Final dataset size after deduplication: {len(loader.data)}")
print(f"  - Non-Offensive (0): {sum(loader.data['label'] == 0)}")
print(f"  - Offensive     (1): {sum(loader.data['label'] == 1)}")

# Save the new processed data
loader.save_processed_data()

# ─────────────────────────────────────────────
# STEP 2: Train / Val / Test Split
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Splitting Data (70/10/20)")
print("="*60)

# Binary: map all non-zero labels to 1 (offensive)
loader.data['label'] = loader.data['label'].apply(lambda x: 1 if x > 0 else 0)

train_val, test = train_test_split(
    loader.data, test_size=0.2, random_state=42, stratify=loader.data['label']
)
train, val = train_test_split(
    train_val, test_size=0.125, random_state=42, stratify=train_val['label']
)

print(f"  Train  : {len(train)} samples")
print(f"  Val    : {len(val)}   samples")
print(f"  Test   : {len(test)}  samples")

# ─────────────────────────────────────────────
# STEP 3: TF-IDF Feature Extraction
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: TF-IDF Feature Extraction")
print("="*60)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
X_train_tfidf = tfidf.fit_transform(train['text'])
X_val_tfidf   = tfidf.transform(val['text'])
X_test_tfidf  = tfidf.transform(test['text'])

y_train = train['label'].values
y_val   = val['label'].values
y_test  = test['label'].values

print(f"  ✓ TF-IDF shape: {X_train_tfidf.shape}")

# Save TF-IDF vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print("  ✓ Saved TF-IDF vectorizer")

# ─────────────────────────────────────────────
# Helper: Print & return metrics
# ─────────────────────────────────────────────
def evaluate(name, model, X_t, y_t, X_v, y_v, X_te, y_te, is_dense=False):
    if is_dense:
        X_t  = X_t.toarray()
        X_v  = X_v.toarray()
        X_te = X_te.toarray()

    model.fit(X_t, y_t)
    y_pred_val  = model.predict(X_v)
    y_pred_test = model.predict(X_te)

    val_acc  = accuracy_score(y_v,  y_pred_val)
    test_acc = accuracy_score(y_te, y_pred_test)
    prec     = precision_score(y_te, y_pred_test, zero_division=0)
    rec      = recall_score(y_te,    y_pred_test, zero_division=0)
    f1       = f1_score(y_te,        y_pred_test, zero_division=0)

    print(f"\n  [{name}]")
    print(f"    Val  Accuracy : {val_acc:.4f}")
    print(f"    Test Accuracy : {test_acc:.4f}")
    print(f"    Precision     : {prec:.4f}")
    print(f"    Recall        : {rec:.4f}")
    print(f"    F1-Score      : {f1:.4f}")
    print(classification_report(y_te, y_pred_test,
                                target_names=['Non-Offensive', 'Offensive']))
    return model, {"val_acc": val_acc, "test_acc": test_acc,
                   "precision": prec, "recall": rec, "f1": f1}

# ─────────────────────────────────────────────
# STEP 4: Train ML Models
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: Training ML Models (SVM / NB / RF)")
print("="*60)

results = {}

# SVM
svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm, r = evaluate("SVM", svm,
                  X_train_tfidf, y_train,
                  X_val_tfidf,   y_val,
                  X_test_tfidf,  y_test)
results['SVM'] = r
with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
    pickle.dump(svm, f)
print("  ✓ Saved SVM model")

# Naive Bayes
nb = MultinomialNB(alpha=1.0)
nb, r = evaluate("Naive Bayes", nb,
                 X_train_tfidf, y_train,
                 X_val_tfidf,   y_val,
                 X_test_tfidf,  y_test)
results['Naive_Bayes'] = r
with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(nb, f)
print("  ✓ Saved Naive Bayes model")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                             min_samples_split=5, random_state=42, n_jobs=-1)
rf, r = evaluate("Random Forest", rf,
                 X_train_tfidf, y_train,
                 X_val_tfidf,   y_val,
                 X_test_tfidf,  y_test,
                 is_dense=True)
results['Random_Forest'] = r
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(rf, f)
print("  ✓ Saved Random Forest model")

# ─────────────────────────────────────────────
# STEP 5: Train Deep Learning Models (CNN + LSTM)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5: Training Deep Learning Models (CNN + LSTM)")
print("="*60)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D,
                                         Dense, LSTM, Dropout, SpatialDropout1D)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    MAX_LEN       = 100
    MAX_WORDS     = 20000
    EMBEDDING_DIM = 100
    EPOCHS        = 10
    BATCH_SIZE    = 64
    NUM_CLASSES   = 2   # binary: offensive / non-offensive

    # Tokenize
    print("  Fitting tokenizer ...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train['text'])

    def to_seq(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

    X_tr_seq  = to_seq(train['text'])
    X_va_seq  = to_seq(val['text'])
    X_te_seq  = to_seq(test['text'])

    y_tr_cat  = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_va_cat  = to_categorical(y_val,   num_classes=NUM_CLASSES)
    y_te_cat  = to_categorical(y_test,  num_classes=NUM_CLASSES)

    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"  Vocab size : {vocab_size}")

    # Save tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print("  ✓ Saved tokenizer")

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # ── CNN ──
    print("\n  [CNN] Building & Training ...")
    cnn = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_tr_seq, y_tr_cat,
            validation_data=(X_va_seq, y_va_cat),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[early_stop], verbose=1)

    cnn_loss, cnn_acc = cnn.evaluate(X_te_seq, y_te_cat, verbose=0)
    print(f"  CNN  Test Accuracy: {cnn_acc:.4f}")
    results['CNN'] = {"test_acc": cnn_acc}
    cnn.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
    print("  ✓ Saved CNN model")

    # ── LSTM ──
    print("\n  [LSTM] Building & Training ...")
    lstm = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_tr_seq, y_tr_cat,
             validation_data=(X_va_seq, y_va_cat),
             epochs=EPOCHS, batch_size=BATCH_SIZE,
             callbacks=[early_stop], verbose=1)

    lstm_loss, lstm_acc = lstm.evaluate(X_te_seq, y_te_cat, verbose=0)
    print(f"  LSTM Test Accuracy: {lstm_acc:.4f}")
    results['LSTM'] = {"test_acc": lstm_acc}
    lstm.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    print("  ✓ Saved LSTM model")

except Exception as e:
    print(f"\n  [WARNING] Deep learning skipped: {e}")
    print("  (Install TensorFlow to enable CNN/LSTM training)")

# ─────────────────────────────────────────────
# STEP 6: Final Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL RESULTS — ALL MODELS")
print("="*60)
print(f"  {'Model':<18} {'Val Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<10} {'F1'}")
print("  " + "-"*68)
for name, r in results.items():
    va  = f"{r.get('val_acc', 0):.4f}"
    ta  = f"{r.get('test_acc', 0):.4f}"
    pr  = f"{r.get('precision', 0):.4f}"
    rc  = f"{r.get('recall', 0):.4f}"
    f1  = f"{r.get('f1', 0):.4f}"
    print(f"  {name:<18} {va:<12} {ta:<12} {pr:<12} {rc:<10} {f1}")

print("\n✅ All models retrained on 11 datasets (including T1, T2, T3)!")
print(f"   Models saved to: {os.path.abspath(MODEL_DIR)}/")
