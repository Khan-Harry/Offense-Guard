"""
Training ML and DL Models Specifically on Offensive-24K-T1 (Offense Detection) Dataset
Models:
  - ML: SVM, Naive Bayes, Random Forest (with TF-IDF Vectorizer)
  - DL: CNN, LSTM (with Tokenizer and Word2Vec Embeddings)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)

# Set utf-8 stdout if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATASET_FILE = "Offensive-24K-T1(Offense Detection).xlsx"

def clean_text(text):
    """Clean and normalize Urdu/Roman Urdu text"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+|USER', '', text, flags=re.IGNORECASE)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("="*60)
print(" STEP 1: Loading & Preprocessing Offensive-24K-T1 Dataset")
print("="*60)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"File {DATASET_FILE} not found!")

df = pd.read_excel(DATASET_FILE)
print(f"Loaded raw dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure correct column names
text_col = 'Tweet' if 'Tweet' in df.columns else df.columns[1]
tag_col = 'Tag' if 'Tag' in df.columns else df.columns[2]

df['text'] = df[text_col].apply(clean_text)

def parse_tag(x):
    if isinstance(x, str):
        x = x.strip().upper()
        if x == 'OFF' or x == '1':
            return 1
        elif x == 'NOT' or x == '0':
            return 0
    try:
        val = int(x)
        return 1 if val == 1 else 0
    except:
        return 0

df['label'] = df[tag_col].apply(parse_tag)

# Drop empty / na
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.len() > 1]
initial_len = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"Removed {initial_len - len(df)} duplicate/empty rows.")
print(f"Total clean samples: {len(df)}")
print(f"  - Non-Offensive (0): {sum(df['label'] == 0)}")
print(f"  - Offensive     (1): {sum(df['label'] == 1)}")

# ─────────────────────────────────────────────
# STEP 2: Train / Val / Test Split (70/10/20)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Splitting Data (70% Train, 10% Val, 20% Test)")
print("="*60)

train_val, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])

print(f"Train samples: {len(train_df)}")
print(f"Val samples  : {len(val_df)}")
print(f"Test samples : {len(test_df)}")

# ─────────────────────────────────────────────
# STEP 3: TF-IDF Feature Extraction
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: TF-IDF Vectorization for ML Models")
print("="*60)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.85)
X_train_tfidf = tfidf.fit_transform(train_df['text'])
X_val_tfidf   = tfidf.transform(val_df['text'])
X_test_tfidf  = tfidf.transform(test_df['text'])

y_train = train_df['label'].values
y_val   = val_df['label'].values
y_test  = test_df['label'].values

print(f"TF-IDF Shape: {X_train_tfidf.shape}")
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print("Saved models/tfidf_vectorizer.pkl")

results = {}

# ─────────────────────────────────────────────
# STEP 4: Train Traditional ML Models
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 4: Training Machine Learning Models (SVM, NB, RF)")
print("="*60)

# 1. SVM
print("\n[1/3] Training Linear SVM...")
svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred_svm_val = svm.predict(X_val_tfidf)
y_pred_svm_test = svm.predict(X_test_tfidf)

svm_val_acc = accuracy_score(y_val, y_pred_svm_val)
svm_test_acc = accuracy_score(y_test, y_pred_svm_test)
svm_prec = precision_score(y_test, y_pred_svm_test, zero_division=0)
svm_rec = recall_score(y_test, y_pred_svm_test, zero_division=0)
svm_f1 = f1_score(y_test, y_pred_svm_test, zero_division=0)

results['SVM'] = {
    'val_acc': svm_val_acc, 'test_acc': svm_test_acc,
    'precision': svm_prec, 'recall': svm_rec, 'f1': svm_f1
}
print(f"SVM Test Accuracy: {svm_test_acc:.4f} | F1: {svm_f1:.4f} | Precision: {svm_prec:.4f} | Recall: {svm_rec:.4f}")
with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
    pickle.dump(svm, f)
print("Saved models/svm.pkl")

# 2. Naive Bayes
print("\n[2/3] Training Multinomial Naive Bayes...")
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
y_pred_nb_val = nb.predict(X_val_tfidf)
y_pred_nb_test = nb.predict(X_test_tfidf)

nb_val_acc = accuracy_score(y_val, y_pred_nb_val)
nb_test_acc = accuracy_score(y_test, y_pred_nb_test)
nb_prec = precision_score(y_test, y_pred_nb_test, zero_division=0)
nb_rec = recall_score(y_test, y_pred_nb_test, zero_division=0)
nb_f1 = f1_score(y_test, y_pred_nb_test, zero_division=0)

results['Naive_Bayes'] = {
    'val_acc': nb_val_acc, 'test_acc': nb_test_acc,
    'precision': nb_prec, 'recall': nb_rec, 'f1': nb_f1
}
print(f"NB Test Accuracy: {nb_test_acc:.4f} | F1: {nb_f1:.4f} | Precision: {nb_prec:.4f} | Recall: {nb_rec:.4f}")
with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(nb, f)
print("Saved models/naive_bayes.pkl")

# 3. Random Forest
print("\n[3/3] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
y_pred_rf_val = rf.predict(X_val_tfidf)
y_pred_rf_test = rf.predict(X_test_tfidf)

rf_val_acc = accuracy_score(y_val, y_pred_rf_val)
rf_test_acc = accuracy_score(y_test, y_pred_rf_test)
rf_prec = precision_score(y_test, y_pred_rf_test, zero_division=0)
rf_rec = recall_score(y_test, y_pred_rf_test, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf_test, zero_division=0)

results['Random_Forest'] = {
    'val_acc': rf_val_acc, 'test_acc': rf_test_acc,
    'precision': rf_prec, 'recall': rf_rec, 'f1': rf_f1
}
print(f"RF Test Accuracy: {rf_test_acc:.4f} | F1: {rf_f1:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(rf, f)
print("Saved models/random_forest.pkl")

# ─────────────────────────────────────────────
# STEP 5: Word2Vec + Deep Learning (CNN & LSTM)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 5: Training Word2Vec & Deep Learning Models (CNN, LSTM)")
print("="*60)

try:
    from gensim.models import Word2Vec
    print("Training Word2Vec model on dataset...")
    tokenized_sentences = [text.split() for text in train_df['text']]
    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
    w2v_model.save(os.path.join(MODEL_DIR, "word2vec.model"))
    print("Saved models/word2vec.model")
except Exception as e:
    print(f"Word2Vec notice: {e}")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    MAX_LEN = 100
    MAX_WORDS = 20000
    EMBEDDING_DIM = 100
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_CLASSES = 2

    print("Fitting DL Tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['text'])

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved models/tokenizer.pkl")

    def to_padded_seq(texts):
        seq = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    X_tr_seq = to_padded_seq(train_df['text'])
    X_va_seq = to_padded_seq(val_df['text'])
    X_te_seq = to_padded_seq(test_df['text'])

    y_tr_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_va_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_te_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)

    # Embedding Matrix from Word2Vec if available
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    if 'w2v_model' in locals():
        for word, i in tokenizer.word_index.items():
            if i < vocab_size:
                if word in w2v_model.wv:
                    embedding_matrix[i] = w2v_model.wv[word]
                else:
                    embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # 1. CNN Model
    print("\n--- Training CNN Architecture ---")
    cnn_model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=True),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_tr_seq, y_tr_cat, validation_data=(X_va_seq, y_va_cat), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

    y_pred_cnn = np.argmax(cnn_model.predict(X_te_seq), axis=1)
    cnn_test_acc = accuracy_score(y_test, y_pred_cnn)
    cnn_f1 = f1_score(y_test, y_pred_cnn, zero_division=0)
    cnn_prec = precision_score(y_test, y_pred_cnn, zero_division=0)
    cnn_rec = recall_score(y_test, y_pred_cnn, zero_division=0)

    results['CNN'] = {'test_acc': cnn_test_acc, 'f1': cnn_f1, 'precision': cnn_prec, 'recall': cnn_rec}
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f} | F1: {cnn_f1:.4f}")
    cnn_model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
    print("Saved models/cnn_model.h5")

    # 2. LSTM Model
    print("\n--- Training LSTM Architecture ---")
    lstm_model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=True),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_tr_seq, y_tr_cat, validation_data=(X_va_seq, y_va_cat), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

    y_pred_lstm = np.argmax(lstm_model.predict(X_te_seq), axis=1)
    lstm_test_acc = accuracy_score(y_test, y_pred_lstm)
    lstm_f1 = f1_score(y_test, y_pred_lstm, zero_division=0)
    lstm_prec = precision_score(y_test, y_pred_lstm, zero_division=0)
    lstm_rec = recall_score(y_test, y_pred_lstm, zero_division=0)

    results['LSTM'] = {'test_acc': lstm_test_acc, 'f1': lstm_f1, 'precision': lstm_prec, 'recall': lstm_rec}
    print(f"LSTM Test Accuracy: {lstm_test_acc:.4f} | F1: {lstm_f1:.4f}")
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    print("Saved models/lstm_model.h5")

except Exception as e:
    print(f"\n[Deep Learning Step Notice]: {e}")

# ─────────────────────────────────────────────
# STEP 6: Summary Table
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("  EVALUATION SUMMARY ON OFFENSIVE-24K-T1 TEST SET")
print("="*70)
print(f"  {'Model':<16} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
print("  " + "-"*64)
for name, m in results.items():
    ta = f"{m.get('test_acc', 0):.4f}"
    pr = f"{m.get('precision', 0):.4f}"
    rc = f"{m.get('recall', 0):.4f}"
    f1 = f"{m.get('f1', 0):.4f}"
    print(f"  {name:<16} {ta:<12} {pr:<12} {rc:<12} {f1:<10}")

print("\n" + "="*70)
print("Training completed successfully on Offensive-24K-T1!")
print("="*70)
