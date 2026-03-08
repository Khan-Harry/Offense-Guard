"""
Feature Extraction Module
Implements TF-IDF, Bag of Words, and Word Embeddings for text classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import pickle
import os


class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        self.word2vec_model = None
        
    def fit_tfidf(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Fit TF-IDF vectorizer
        
        TF-IDF (Term Frequency-Inverse Document Frequency):
        - Measures importance of words in documents
        - TF: How often a word appears in a document
        - IDF: How rare a word is across all documents
        - High TF-IDF = word is frequent in this doc but rare overall
        
        Why use it?
        - Captures word importance better than simple word counts
        - Reduces impact of common words (like "the", "is")
        - Works well with traditional ML models (SVM, Naive Bayes)
        """
        print("\n=== Training TF-IDF Vectorizer ===")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,  # Unigrams and bigrams
            min_df=2,  # Ignore words appearing in less than 2 documents
            max_df=0.8  # Ignore words appearing in more than 80% of documents
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(texts)
        print(f"✓ TF-IDF features shape: {X_tfidf.shape}")
        print(f"  - Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return X_tfidf
    
    def transform_tfidf(self, texts):
        """Transform texts using fitted TF-IDF vectorizer"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted! Call fit_tfidf() first.")
        return self.tfidf_vectorizer.transform(texts)
    
    def fit_bow(self, texts, max_features=5000):
        """
        Fit Bag of Words vectorizer
        
        Bag of Words (BoW):
        - Represents text as word frequency counts
        - Each document becomes a vector of word counts
        - Ignores word order and grammar
        
        Why use it?
        - Simple and interpretable
        - Fast to compute
        - Good baseline for text classification
        - Works well with Naive Bayes
        """
        print("\n=== Training Bag of Words Vectorizer ===")
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8
        )
        
        X_bow = self.bow_vectorizer.fit_transform(texts)
        print(f"✓ BoW features shape: {X_bow.shape}")
        print(f"  - Vocabulary size: {len(self.bow_vectorizer.vocabulary_)}")
        
        return X_bow
    
    def transform_bow(self, texts):
        """Transform texts using fitted BoW vectorizer"""
        if self.bow_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted! Call fit_bow() first.")
        return self.bow_vectorizer.transform(texts)
    
    def fit_word2vec(self, texts, vector_size=100, window=5, min_count=2):
        """
        Train Word2Vec model
        
        Word2Vec:
        - Creates dense vector representations of words
        - Words with similar meanings have similar vectors
        - Captures semantic relationships (e.g., "king" - "man" + "woman" ≈ "queen")
        
        Why use it?
        - Captures word semantics and context
        - Handles synonyms and related words better
        - Works well with deep learning models (CNN, LSTM)
        - Reduces dimensionality compared to TF-IDF
        """
        print("\n=== Training Word2Vec Model ===")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # Skip-gram model (better for small datasets)
        )
        
        print(f"✓ Word2Vec model trained")
        print(f"  - Vocabulary size: {len(self.word2vec_model.wv)}")
        print(f"  - Vector size: {vector_size}")
        
        return self.word2vec_model
    
    def text_to_word2vec(self, text):
        """Convert text to Word2Vec vector by averaging word vectors"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained! Call fit_word2vec() first.")
        
        words = text.split()
        word_vectors = []
        
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        if len(word_vectors) == 0:
            # Return zero vector if no words found
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average all word vectors
        return np.mean(word_vectors, axis=0)
    
    def texts_to_word2vec(self, texts):
        """Convert multiple texts to Word2Vec vectors"""
        return np.array([self.text_to_word2vec(text) for text in texts])
    
    def save_models(self, output_dir="models"):
        """Save all fitted models"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.tfidf_vectorizer:
            with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"✓ Saved TF-IDF vectorizer")
        
        if self.bow_vectorizer:
            with open(os.path.join(output_dir, "bow_vectorizer.pkl"), "wb") as f:
                pickle.dump(self.bow_vectorizer, f)
            print(f"✓ Saved BoW vectorizer")
        
        if self.word2vec_model:
            self.word2vec_model.save(os.path.join(output_dir, "word2vec.model"))
            print(f"✓ Saved Word2Vec model")
    
    def load_models(self, output_dir="models"):
        """Load all saved models"""
        tfidf_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"✓ Loaded TF-IDF vectorizer")
        
        bow_path = os.path.join(output_dir, "bow_vectorizer.pkl")
        if os.path.exists(bow_path):
            with open(bow_path, "rb") as f:
                self.bow_vectorizer = pickle.load(f)
            print(f"✓ Loaded BoW vectorizer")
        
        w2v_path = os.path.join(output_dir, "word2vec.model")
        if os.path.exists(w2v_path):
            self.word2vec_model = Word2Vec.load(w2v_path)
            print(f"✓ Loaded Word2Vec model")

    def prepare_sequences(self, texts, max_len=100, max_words=10000):
        """
        Prepare sequences for Deep Learning (CNN/LSTM)
        - Tokenizes text into sequences of integers
        - Pads sequences to uniform length
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        print(f"\n=== Preparing Sequences (Max Len: {max_len}) ===")
        
        tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        print(f"✓ Prepared {len(padded)} sequences")
        return padded, tokenizer

    def get_embedding_matrix(self, tokenizer, embedding_dim=100):
        """Create embedding matrix from Word2Vec for Keras Embedding layer"""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained! Call fit_word2vec() first.")
        
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        for word, i in tokenizer.word_index.items():
            if word in self.word2vec_model.wv:
                embedding_matrix[i] = self.word2vec_model.wv[word]
                
        print(f"✓ Created embedding matrix of shape: {embedding_matrix.shape}")
        return embedding_matrix, vocab_size


# Example usage
if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv("d:/Semesters/BSE-6/FYP 2/FYP_Project/processed_data.csv")
    
    # Split data (same as in data_loader.py)
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])
    
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Fit and transform TF-IDF
    X_train_tfidf = extractor.fit_tfidf(train['text'])
    X_val_tfidf = extractor.transform_tfidf(val['text'])
    X_test_tfidf = extractor.transform_tfidf(test['text'])
    
    # Fit and transform BoW
    X_train_bow = extractor.fit_bow(train['text'])
    X_val_bow = extractor.transform_bow(val['text'])
    X_test_bow = extractor.transform_bow(test['text'])
    
    # Fit and transform Word2Vec
    extractor.fit_word2vec(train['text'])
    X_train_w2v = extractor.texts_to_word2vec(train['text'])
    X_val_w2v = extractor.texts_to_word2vec(val['text'])
    X_test_w2v = extractor.texts_to_word2vec(test['text'])
    
    print(f"\n=== Feature Shapes ===")
    print(f"TF-IDF: Train {X_train_tfidf.shape}, Val {X_val_tfidf.shape}, Test {X_test_tfidf.shape}")
    print(f"BoW: Train {X_train_bow.shape}, Val {X_val_bow.shape}, Test {X_test_bow.shape}")
    print(f"Word2Vec: Train {X_train_w2v.shape}, Val {X_val_w2v.shape}, Test {X_test_w2v.shape}")
    
    # Save models
    extractor.save_models()
