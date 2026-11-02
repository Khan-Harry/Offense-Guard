# -*- coding: utf-8 -*-
"""
CNN & Bi-LSTM Retraining Script — Keras 3.x Compatible
=======================================================
Retrains CNN and Bi-LSTM models using existing tokenizer + dataset,
saves new Keras 3.x compatible .keras files (replaces old .h5 models).

Usage:
    .\.venv\Scripts\python.exe retrain_cnn_lstm.py

Output:
    models/cnn_model.keras   (Keras 3.x compatible)
    models/lstm_model.keras  (Keras 3.x compatible)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import pickle

print("=" * 65)
print(" OFFENSEGUARD — CNN & Bi-LSTM RETRAIN (Keras 3.x)")
print("=" * 65)

# ── 1. Load existing tokenizer ──────────────────────────────────────
print("\n[1/6] Loading existing tokenizer...")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print(f"    ✓ Tokenizer loaded — vocab size: {len(tokenizer.word_index):,}")

# ── 2. Load dataset ─────────────────────────────────────────────────
print("\n[2/6] Loading dataset...")
df = pd.read_csv("processed_data.csv")
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int).clip(0, 1)   # ensure binary
print(f"    ✓ Dataset loaded — {len(df):,} rows")
print(f"    Offensive: {df['label'].sum():,} | Safe: {(df['label']==0).sum():,}")

# ── 3. Prepare sequences ─────────────────────────────────────────────
print("\n[3/6] Preparing padded sequences...")
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

MAXLEN = 100

seqs = tokenizer.texts_to_sequences(df['text'].tolist())
X = pad_sequences(seqs, maxlen=MAXLEN, padding='post', truncating='post')
y = df['label'].values

# Stratified split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"    ✓ Train: {len(X_train):,} | Val: {len(X_val):,}")

# ── 4. Build & Train CNN ─────────────────────────────────────────────
print("\n[4/6] Building 1D CNN model (Keras 3.x)...")
import keras
from keras import layers

VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBED_DIM = 128

def build_cnn(vocab_size, maxlen, embed_dim):
    inputs = keras.Input(shape=(maxlen,), name="cnn_input")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)
    x = layers.Conv1D(128, 3, activation='relu', padding='same', name="conv1d")(x)
    x = layers.GlobalMaxPooling1D(name="global_pool")(x)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(2, activation='softmax', name="output")(x)
    model = keras.Model(inputs, outputs, name="cnn_offenseguard")
    return model

cnn_model = build_cnn(VOCAB_SIZE, MAXLEN, EMBED_DIM)
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
cnn_model.summary()

print("\n    Training CNN...")
t0 = time.time()
cnn_history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, verbose=1)
    ],
    verbose=1
)
cnn_elapsed = time.time() - t0

# Evaluate
cnn_val_acc = max(cnn_history.history['val_accuracy'])
print(f"\n    ✓ CNN Training complete in {cnn_elapsed:.1f}s")
print(f"    Best Val Accuracy: {cnn_val_acc*100:.2f}%")

# Save
CNN_SAVE_PATH = os.path.join("models", "cnn_model.keras")
cnn_model.save(CNN_SAVE_PATH)
print(f"    ✓ Saved: {CNN_SAVE_PATH}")

# ── 5. Build & Train Bi-LSTM ─────────────────────────────────────────
print("\n[5/6] Building Bi-LSTM model (Keras 3.x)...")

def build_bilstm(vocab_size, maxlen, embed_dim):
    inputs = keras.Input(shape=(maxlen,), name="lstm_input")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, name="lstm"),
        name="bidirectional"
    )(x)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(2, activation='softmax', name="output")(x)
    model = keras.Model(inputs, outputs, name="bilstm_offenseguard")
    return model

lstm_model = build_bilstm(VOCAB_SIZE, MAXLEN, EMBED_DIM)
lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
lstm_model.summary()

print("\n    Training Bi-LSTM...")
t0 = time.time()
lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, verbose=1)
    ],
    verbose=1
)
lstm_elapsed = time.time() - t0

lstm_val_acc = max(lstm_history.history['val_accuracy'])
print(f"\n    ✓ Bi-LSTM Training complete in {lstm_elapsed:.1f}s")
print(f"    Best Val Accuracy: {lstm_val_acc*100:.2f}%")

LSTM_SAVE_PATH = os.path.join("models", "lstm_model.keras")
lstm_model.save(LSTM_SAVE_PATH)
print(f"    ✓ Saved: {LSTM_SAVE_PATH}")

# ── 6. Final Summary ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" RETRAINING COMPLETE")
print("=" * 65)
print(f"  CNN  Best Val Accuracy  : {cnn_val_acc*100:.2f}%")
print(f"  LSTM Best Val Accuracy  : {lstm_val_acc*100:.2f}%")
print(f"  Files saved to          : models/")
print(f"  Keras version used      : {keras.__version__}")
print("\n  [NEXT] Restart Flask server: python app.py")
print("=" * 65)
