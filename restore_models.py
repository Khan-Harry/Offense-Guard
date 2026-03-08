import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

print("=== Restoring SVM & Naive Bayes Models ===")

# Load processed data
data = pd.read_csv("processed_data.csv")
print(f"Loaded {len(data)} samples")

# Extract features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
X = tfidf.fit_transform(data['text'])
y = data['label'].values

# Train Naive Bayes
print("Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X, y)

# Train SVM
print("Training SVM...")
svm = LinearSVC(C=1.0, random_state=42)
svm.fit(X, y)

# Save
os.makedirs("models", exist_ok=True)
with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(nb, f)
with open("models/svm.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✓ Models restored successfully!")
