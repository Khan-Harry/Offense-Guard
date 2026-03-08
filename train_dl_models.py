"""
Train Deep Learning Models (CNN and LSTM) for Multi-class Text Classification
"""

import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Dropout, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from data_loader import DataLoader
from feature_extraction import FeatureExtractor

# Configuration
MAX_LEN = 100
MAX_WORDS = 20000
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 64

def build_cnn_model(vocab_size, embedding_matrix, num_classes):
    """Convolutional Neural Network (CNN) for text classification"""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], 
                  input_length=MAX_LEN, trainable=True),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size, embedding_matrix, num_classes):
    """Long Short-Term Memory (LSTM) for text classification"""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], 
                  input_length=MAX_LEN, trainable=True),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Load Data
    loader = DataLoader()
    loader.load_datasets()
    data = loader.preprocess()
    train_df, val_df, test_df = loader.split_data()
    
    # 2. Extract Features
    extractor = FeatureExtractor()
    
    # Fit Word2Vec on train data
    extractor.fit_word2vec(train_df['text'], vector_size=EMBEDDING_DIM)
    
    # Prepare sequences
    X_train, tokenizer = extractor.prepare_sequences(train_df['text'], max_len=MAX_LEN, max_words=MAX_WORDS)
    
    # Transform val and test texts using the same tokenizer
    val_sequences = tokenizer.texts_to_sequences(val_df['text'])
    X_val = pad_sequences(val_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    test_sequences = tokenizer.texts_to_sequences(test_df['text'])
    X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Get embedding matrix
    embedding_matrix, vocab_size = extractor.get_embedding_matrix(tokenizer, embedding_dim=EMBEDDING_DIM)
    
    # 3. Prepare Labels (One-Hot Encoding)
    num_classes = 4  # Neutral, Hate Speech, Abusive, Offensive
    y_train = to_categorical(train_df['label'], num_classes=num_classes)
    y_val = to_categorical(val_df['label'], num_classes=num_classes)
    y_test = to_categorical(test_df['label'], num_classes=num_classes)
    
    # 4. Train Models
    print("\n--- Training CNN Model ---")
    cnn_model = build_cnn_model(vocab_size, embedding_matrix, num_classes)
    cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("\n--- Training LSTM Model ---")
    lstm_model = build_lstm_model(vocab_size, embedding_matrix, num_classes)
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                   epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 5. Evaluate Models
    print("\n--- CNN Evaluation ---")
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test)
    print(f"CNN Accuracy: {cnn_acc:.4f}")
    
    print("\n--- LSTM Evaluation ---")
    lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
    
    # 6. Save Models
    os.makedirs("models", exist_ok=True)
    cnn_model.save("models/cnn_model.h5")
    lstm_model.save("models/lstm_model.h5")
    
    # Save tokenizer
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
        
    print("\n✓ Models and tokenizer saved to 'models/' directory")

# Helper to use tokenizer transforms in main
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    # Small fix for the sequence prep logic in main
    # Actually, let's redefine pad_sequences inside the script to avoid import confusion
    main()
