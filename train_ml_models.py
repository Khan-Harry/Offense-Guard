"""
Machine Learning Models Training
Trains and evaluates Naive Bayes, SVM, and Random Forest classifiers
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_naive_bayes(self, X_train, y_train, X_val, y_val):
        """
        Train Naive Bayes classifier
        
        Naive Bayes:
        - Probabilistic classifier based on Bayes' theorem
        - Assumes features are independent (naive assumption)
        - Fast training and prediction
        - Works well with text data and TF-IDF/BoW features
        
        Why use it?
        - Excellent baseline for text classification
        - Handles high-dimensional sparse data well
        - Requires little training data
        - Interpretable probabilities
        """
        print("\n=== Training Naive Bayes ===")
        
        nb_model = MultinomialNB(alpha=1.0)  # Laplace smoothing
        nb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = nb_model.predict(X_train)
        y_pred_val = nb_model.predict(X_val)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val)
        val_recall = recall_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val)
        
        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Validation Accuracy: {val_acc:.4f}")
        print(f"✓ Precision: {val_precision:.4f}")
        print(f"✓ Recall: {val_recall:.4f}")
        print(f"✓ F1-Score: {val_f1:.4f}")
        
        self.models['naive_bayes'] = nb_model
        self.results['naive_bayes'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'y_pred_val': y_pred_val
        }
        
        return nb_model
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """
        Train Support Vector Machine classifier
        
        SVM (Support Vector Machine):
        - Finds optimal hyperplane to separate classes
        - Maximizes margin between classes
        - Effective in high-dimensional spaces
        - Works well with text classification
        
        Why use it?
        - High accuracy on text data
        - Robust to overfitting (especially with regularization)
        - Works well with TF-IDF features
        - Good generalization
        """
        print("\n=== Training SVM ===")
        
        svm_model = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = svm_model.predict(X_train)
        y_pred_val = svm_model.predict(X_val)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val)
        val_recall = recall_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val)
        
        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Validation Accuracy: {val_acc:.4f}")
        print(f"✓ Precision: {val_precision:.4f}")
        print(f"✓ Recall: {val_recall:.4f}")
        print(f"✓ F1-Score: {val_f1:.4f}")
        
        self.models['svm'] = svm_model
        self.results['svm'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'y_pred_val': y_pred_val
        }
        
        return svm_model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest classifier
        
        Random Forest:
        - Ensemble of decision trees
        - Each tree votes for a class
        - Reduces overfitting through averaging
        - Handles non-linear relationships
        
        Why use it?
        - Robust and accurate
        - Handles feature interactions well
        - Less prone to overfitting than single decision tree
        - Provides feature importance
        """
        print("\n=== Training Random Forest ===")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_val = rf_model.predict(X_val)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val)
        val_recall = recall_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val)
        
        print(f"✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Validation Accuracy: {val_acc:.4f}")
        print(f"✓ Precision: {val_precision:.4f}")
        print(f"✓ Recall: {val_recall:.4f}")
        print(f"✓ F1-Score: {val_f1:.4f}")
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'y_pred_val': y_pred_val
        }
        
        return rf_model
    
    def evaluate_on_test(self, X_test, y_test, model_name):
        """Evaluate a specific model on test set"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained!")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n=== {model_name.upper()} Test Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Offensive', 'Offensive']))
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== Model Comparison (Validation Set) ===")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 68)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} {results['val_acc']:<12.4f} {results['precision']:<12.4f} "
                  f"{results['recall']:<12.4f} {results['f1']:<12.4f}")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        print(f"\n✓ Best Model: {best_model[0].upper()} (F1-Score: {best_model[1]['f1']:.4f})")
        
        return best_model[0]
    
    def save_models(self, output_dir="models"):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✓ Saved {model_name}")
    
    def load_model(self, model_name, model_dir="models"):
        """Load a saved model"""
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        self.models[model_name] = model
        print(f"✓ Loaded {model_name}")
        return model


# Example usage
if __name__ == "__main__":
    # Load processed data
    data = pd.read_csv("d:/Semesters/BSE-6/FYP 2/FYP_Project/processed_data.csv")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Extract features using TF-IDF
    print("\n=== Extracting TF-IDF Features ===")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_train = tfidf.fit_transform(train['text'])
    X_val = tfidf.transform(val['text'])
    X_test = tfidf.transform(test['text'])
    
    y_train = train['label'].values
    y_val = val['label'].values
    y_test = test['label'].values
    
    print(f"✓ Feature shape: {X_train.shape}")
    
    # Train models
    trainer = MLModelTrainer()
    
    trainer.train_naive_bayes(X_train, y_train, X_val, y_val)
    trainer.train_svm(X_train, y_train, X_val, y_val)
    trainer.train_random_forest(X_train.toarray(), y_train, X_val.toarray(), y_val)  # RF needs dense arrays
    
    # Compare models
    best_model_name = trainer.compare_models()
    
    # Evaluate best model on test set
    if best_model_name == 'random_forest':
        test_results = trainer.evaluate_on_test(X_test.toarray(), y_test, best_model_name)
    else:
        test_results = trainer.evaluate_on_test(X_test, y_test, best_model_name)
    
    # Save models
    trainer.save_models()
    
    # Save TF-IDF vectorizer
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("✓ Saved TF-IDF vectorizer")
