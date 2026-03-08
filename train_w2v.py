"""
Train Word2Vec Embeddings for Urdu/Roman Urdu
This script does not depend on TensorFlow and can be run while TF is installing.
"""

import pandas as pd
import numpy as np
import os
import pickle
from data_loader import DataLoader
from feature_extraction import FeatureExtractor

# Configuration
EMBEDDING_DIM = 100

def main():
    print("\n--- Starting Word2Vec Training ---")
    
    # 1. Load Data
    loader = DataLoader()
    loader.load_datasets()
    data = loader.preprocess()
    train_df, val_df, test_df = loader.split_data()
    
    # 2. Initialize Feature Extractor
    extractor = FeatureExtractor()
    
    # 3. Train Word2Vec on train data
    print(f"Training Word2Vec model with dimension {EMBEDDING_DIM}...")
    extractor.fit_word2vec(train_df['text'], vector_size=EMBEDDING_DIM)
    
    # 4. Save Models
    os.makedirs("models", exist_ok=True)
    extractor.save_models("models")
    
    print("\n✓ Word2Vec model trained and saved to 'models/' directory")
    
    # 5. Quick verification
    print("\nVerifying similar words (Roman Urdu example 'hai'):")
    if 'hai' in extractor.word2vec_model.wv:
        print(extractor.word2vec_model.wv.most_similar('hai', topn=5))
    else:
        print("'hai' not in vocabulary")

if __name__ == "__main__":
    main()
