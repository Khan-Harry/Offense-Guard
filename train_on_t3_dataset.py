"""
Training All 5 ML and DL Models Specifically on Offensive-24K-T3 (Target Type Classification) Dataset
Classes:
  - 1: Individual (IND)
  - 2: Group (GRP)
  - 3: Other / Organization (OTH)
Models:
  - ML: Linear SVM (Calibrated), Multinomial Naive Bayes, Random Forest
  - DL: 1D CNN, Bidirectional LSTM
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D,
                                     Dense, LSTM, Dropout, SpatialDropout1D, Bidirectional)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Set utf-8 stdout if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_PATH = "d:/Semesters/BSE-6/FYP 2/FYP_Project"
MODEL_DIR = os.path.join(BASE_PATH, "models_t3")
os.makedirs(MODEL_DIR, exist_ok=True)
DATASET_FILE = os.path.join(BASE_PATH, "Offensive-24K-T3(Target Type Classification).xlsx")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|USER|#', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("="*70)
print(" STEP 1: Loading & Preprocessing Offensive-24K-T3 Dataset")
print("="*70)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"File {DATASET_FILE} not found!")

df = pd.read_excel(DATASET_FILE)
print(f"Loaded raw dataset shape: {df.shape}")

text_col = 'Tweet' if 'Tweet' in df.columns else df.columns[1]
tag_col = 'Tag' if 'Tag' in df.columns else df.columns[2]

df['text'] = df[text_col].apply(clean_text)

def parse_tag(x):
    # Map 1 -> 0 (Individual), 2 -> 1 (Group), 3 -> 2 (Other)
    try:
        val = int(x)
        if val in [1, 2, 3]:
            return val - 1
    except:
        pass
    if isinstance(x, str):
        x = x.strip().upper()
        if x in ['IND', '1', 'INDIVIDUAL']:
            return 0
        elif x in ['GRP', '2', 'GROUP']:
            return 1
        elif x in ['OTH', '3', 'OTHER']:
            return 2
    return 0

df['label'] = df[tag_col].apply(parse_tag)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.len() > 1]
initial_len = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

print(f"Deduplication: Removed {initial_len - len(df)} duplicate/empty rows.")
print(f"Total clean samples: {len(df)}")
print(f"  - Class 0 (Individual): {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
print(f"  - Class 1 (Group)     : {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
print(f"  - Class 2 (Other)     : {sum(df['label'] == 2)} ({sum(df['label'] == 2)/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 2: Train / Val / Test Split (70/10/20)
# ─────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 2: Splitting Data Stratified (70% Train, 10% Val, 20% Test)")
print("="*70)

train_val, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])

print(f"Train samples : {len(train_df)}")
print(f"Val samples   : {len(val_df)}")
print(f"Test samples  : {len(test_df)}")

# ─────────────────────────────────────────────
# STEP 3: TF-IDF Feature Extraction
# ─────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 3: TF-IDF Feature Extraction (Unigram + Bigram + Trigram)")
print("="*70)

tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf   = tfidf.transform(val_df['text'])
X_test_tfidf  = tfidf.transform(test_df['text'])

y_train = train_df['label'].values
y_val   = val_df['label'].values
y_test  = test_df['label'].values

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer_t3.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print("✓ Saved models_t3/tfidf_vectorizer_t3.pkl")

results = {}

# ─────────────────────────────────────────────
# STEP 4: Machine Learning Classifiers
# ─────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 4: Training Machine Learning Classifiers (SVM, NB, RF)")
print("="*70)

# 1. Linear SVM with Calibration
print("\n--- [1/3] Training Calibrated Linear SVM ---")
base_svm = LinearSVC(C=1.0, max_iter=3000, random_state=42)
svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
svm.fit(X_train_tfidf, y_train)

y_pred_svm = svm.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_prec = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_rec = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)

results['SVM'] = {'test_acc': svm_acc, 'f1': svm_f1, 'precision': svm_prec, 'recall': svm_rec}
print(f"SVM Test Accuracy: {svm_acc*100:.2f}% | F1: {svm_f1:.4f} | Precision: {svm_prec:.4f} | Recall: {svm_rec:.4f}")
with open(os.path.join(MODEL_DIR, "svm_t3.pkl"), "wb") as f:
    pickle.dump(svm, f)

# 2. Multinomial Naive Bayes
print("\n--- [2/3] Training Multinomial Naive Bayes ---")
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)
nb_prec = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
nb_rec = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)

results['Naive_Bayes'] = {'test_acc': nb_acc, 'f1': nb_f1, 'precision': nb_prec, 'recall': nb_rec}
print(f"Naive Bayes Test Accuracy: {nb_acc*100:.2f}% | F1: {nb_f1:.4f} | Precision: {nb_prec:.4f} | Recall: {nb_rec:.4f}")
with open(os.path.join(MODEL_DIR, "naive_bayes_t3.pkl"), "wb") as f:
    pickle.dump(nb, f)

# 3. Random Forest
print("\n--- [3/3] Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=120, max_depth=25, min_samples_split=4, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)

y_pred_rf = rf.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_prec = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_rec = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)

results['Random_Forest'] = {'test_acc': rf_acc, 'f1': rf_f1, 'precision': rf_prec, 'recall': rf_rec}
print(f"Random Forest Test Accuracy: {rf_acc*100:.2f}% | F1: {rf_f1:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")
with open(os.path.join(MODEL_DIR, "random_forest_t3.pkl"), "wb") as f:
    pickle.dump(rf, f)

# ─────────────────────────────────────────────
# STEP 5: Deep Learning (CNN & Bi-LSTM)
# ─────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 5: Training Deep Learning Models (1D CNN & Bi-LSTM)")
print("="*70)

MAX_LEN = 100
MAX_WORDS = 20000
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 8

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text'])

with open(os.path.join(MODEL_DIR, "tokenizer_t3.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

def to_seq(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train_seq = to_seq(train_df['text'])
X_val_seq   = to_seq(val_df['text'])
X_test_seq  = to_seq(test_df['text'])

y_train_cat = to_categorical(y_train, num_classes=3)
y_val_cat   = to_categorical(y_val,   num_classes=3)
y_test_cat  = to_categorical(y_test,  num_classes=3)

vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 4. 1D CNN Architecture
print("\n--- [4/5] Training 1D Convolutional Neural Network (CNN) ---")
cnn = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(
    X_train_seq, y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

y_pred_cnn_probs = cnn.predict(X_test_seq)
y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

cnn_acc = accuracy_score(y_test, y_pred_cnn)
cnn_f1 = f1_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
cnn_prec = precision_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
cnn_rec = recall_score(y_test, y_pred_cnn, average='weighted', zero_division=0)

results['CNN'] = {'test_acc': cnn_acc, 'f1': cnn_f1, 'precision': cnn_prec, 'recall': cnn_rec}
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}% | F1: {cnn_f1:.4f} | Precision: {cnn_prec:.4f} | Recall: {cnn_rec:.4f}")
cnn.save(os.path.join(MODEL_DIR, "cnn_model_t3.h5"))

# 5. Bidirectional LSTM Architecture
print("\n--- [5/5] Training Bidirectional LSTM (Bi-LSTM) ---")
lstm = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.25),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm.fit(
    X_train_seq, y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

y_pred_lstm_probs = lstm.predict(X_test_seq)
y_pred_lstm = np.argmax(y_pred_lstm_probs, axis=1)

lstm_acc = accuracy_score(y_test, y_pred_lstm)
lstm_f1 = f1_score(y_test, y_pred_lstm, average='weighted', zero_division=0)
lstm_prec = precision_score(y_test, y_pred_lstm, average='weighted', zero_division=0)
lstm_rec = recall_score(y_test, y_pred_lstm, average='weighted', zero_division=0)

results['LSTM'] = {'test_acc': lstm_acc, 'f1': lstm_f1, 'precision': lstm_prec, 'recall': lstm_rec}
print(f"Bi-LSTM Test Accuracy: {lstm_acc*100:.2f}% | F1: {lstm_f1:.4f} | Precision: {lstm_prec:.4f} | Recall: {lstm_rec:.4f}")
lstm.save(os.path.join(MODEL_DIR, "lstm_model_t3.h5"))

# ─────────────────────────────────────────────
# STEP 6: Summary Table
# ─────────────────────────────────────────────
print("\n" + "="*75)
print("  FINAL EVALUATION RESULTS — ALL 5 MODELS ON OFFENSIVE-24K-T3")
print("="*75)
print(f"  {'Model':<18} {'Test Acc':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<10}")
print("  " + "-"*70)
for name, m in results.items():
    ta = f"{m.get('test_acc', 0)*100:.2f}%"
    pr = f"{m.get('precision', 0)*100:.2f}%"
    rc = f"{m.get('recall', 0)*100:.2f}%"
    f1 = f"{m.get('f1', 0):.4f}"
    print(f"  {name:<18} {ta:<14} {pr:<14} {rc:<14} {f1:<10}")

print("="*75)
print("All 5 Models Trained & Successfully Saved to models_t3/ directory for T3!")
print("="*75)
