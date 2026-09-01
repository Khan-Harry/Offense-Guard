"""
Master Training Script - Context-Aware ML & DL Models
Trained on All 11 Datasets (Urdu + Roman Urdu)
Models Trained:
  1. Linear SVM (Calibrated with Probability)
  2. Multinomial Naive Bayes
  3. Random Forest Classifier
  4. 1D Convolutional Neural Network (CNN)
  5. Bidirectional Long Short-Term Memory (Bi-LSTM)
Features:
  - TF-IDF Vectorizer (Unigrams, Bigrams, Trigrams)
  - End-to-End Deep Word Embeddings (100-dim)
  - Keras Tokenizer (Padded Sequences)
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
                             recall_score, f1_score, classification_report, confusion_matrix)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D,
                                     Dense, LSTM, Dropout, SpatialDropout1D, Bidirectional)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

# Set UTF-8 encoding for terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_PATH = "d:/Semesters/BSE-6/FYP 2/FYP_Project"
MODEL_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1: Text Cleaning & Normalization
# ─────────────────────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @mentions and hashtags markers
    text = re.sub(r'@\w+|USER|#', '', text, flags=re.IGNORECASE)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────────────────────
# STEP 2: Load and Consolidate All 11 Datasets
# ─────────────────────────────────────────────────────────────
print("="*70)
print(" STEP 1: Loading All 11 Datasets with Context-Preserving Labels")
print("="*70)

all_dfs = []

# 1. Hate Speech Roman Urdu (HS-RU-20)
try:
    df1 = pd.read_excel(os.path.join(BASE_PATH, "Hate Speech Roman Urdu (HS-RU-20).xlsx"))
    df1['text'] = df1['Sentence'].apply(clean_text)
    # H = Hostile/Offensive (1), N = Neutral/Non-offensive (0)
    df1['label'] = df1['Neutral (N) / Hostile (H)'].apply(lambda x: 1 if str(x).strip().upper() == 'H' else 0)
    df1 = df1.dropna(subset=['text', 'label'])
    all_dfs.append(df1[['text', 'label']])
    print(f"✓ Loaded HS-RU-20: {len(df1)} samples (Offensive: {sum(df1['label']==1)}, Safe: {sum(df1['label']==0)})")
except Exception as e:
    print(f"✗ Error loading HS-RU-20: {e}")

# 2. Dataset of Urdu Abusive Language
try:
    df2 = pd.read_excel(os.path.join(BASE_PATH, "Dataset of Urdu Abusive Language.xlsx"))
    df2['text'] = df2['no stop'].apply(clean_text)
    # target 1.0 is abusive (1), 0.0 is neutral (0)
    df2['label'] = df2['target'].apply(lambda x: 1 if int(x) == 1 else 0)
    df2 = df2.dropna(subset=['text', 'label'])
    all_dfs.append(df2[['text', 'label']])
    print(f"✓ Loaded Urdu Abusive Language: {len(df2)} samples (Offensive: {sum(df2['label']==1)}, Safe: {sum(df2['label']==0)})")
except Exception as e:
    print(f"✗ Error loading Urdu Abusive dataset: {e}")

# 3. Roman Urdu 30K Dataset
try:
    df3 = None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df3 = pd.read_csv(os.path.join(BASE_PATH, "final 30,000 dataset_romanurdu.csv"), encoding=enc)
            break
        except:
            continue
    if df3 is not None:
        df3['text'] = df3['tweets'].apply(clean_text)
        df3['label'] = df3['label'].apply(lambda x: 1 if str(x).strip().upper() == 'H' else 0)
        df3 = df3.dropna(subset=['text', 'label'])
        all_dfs.append(df3[['text', 'label']])
        print(f"✓ Loaded Roman Urdu 30K: {len(df3)} samples (Offensive: {sum(df3['label']==1)}, Safe: {sum(df3['label']==0)})")
except Exception as e:
    print(f"✗ Error loading 30k dataset: {e}")

# 4. CHate.xlsx (Conversational Hate)
try:
    df_ch = pd.read_excel(os.path.join(BASE_PATH, "CHate.xlsx"))
    df_ch['text'] = df_ch['Roman Urdu'].apply(clean_text)
    df_ch['label'] = 1  # All are offensive/hate speech samples
    df_ch = df_ch.dropna(subset=['text', 'label'])
    all_dfs.append(df_ch[['text', 'label']])
    print(f"✓ Loaded CHate: {len(df_ch)} samples (all offensive)")
except Exception as e:
    print(f"✗ Error loading CHate: {e}")

# 5. GHate.xlsx (Generalized Hate)
try:
    df_gh = pd.read_excel(os.path.join(BASE_PATH, "GHate.xlsx"))
    df_gh['text'] = df_gh['Roman Urdu'].apply(clean_text)
    df_gh['label'] = 1  # All are offensive/hate speech samples
    df_gh = df_gh.dropna(subset=['text', 'label'])
    all_dfs.append(df_gh[['text', 'label']])
    print(f"✓ Loaded GHate: {len(df_gh)} samples (all offensive)")
except Exception as e:
    print(f"✗ Error loading GHate: {e}")

# 6. Cleaned Data (cleaned_data.csv)
try:
    df_cl = pd.read_csv(os.path.join(BASE_PATH, "cleaned_data.csv"))
    df_cl['text'] = df_cl['Comment'].apply(clean_text)
    # Toxic: 1 is safe (0), 0, 2, 3, 4 are offensive (1)
    df_cl['label'] = df_cl['Toxic'].apply(lambda x: 0 if int(x) == 1 else 1)
    df_cl = df_cl.dropna(subset=['text', 'label'])
    all_dfs.append(df_cl[['text', 'label']])
    print(f"✓ Loaded cleaned_data: {len(df_cl)} samples (Offensive: {sum(df_cl['label']==1)}, Safe: {sum(df_cl['label']==0)})")
except Exception as e:
    print(f"✗ Error loading cleaned_data: {e}")

# 7 & 8. task_2_train.csv & task_2_test.csv
for tf_name in ["task_2_train.csv", "task_2_test.csv"]:
    try:
        df_t2_csv = pd.read_csv(os.path.join(BASE_PATH, tf_name), sep='\t', header=None, names=['text', 'label'])
        df_t2_csv['text'] = df_t2_csv['text'].apply(clean_text)
        df_t2_csv['label'] = df_t2_csv['label'].apply(lambda x: 0 if int(x) == 1 else 1)
        df_t2_csv = df_t2_csv.dropna(subset=['text', 'label'])
        all_dfs.append(df_t2_csv[['text', 'label']])
        print(f"✓ Loaded {tf_name}: {len(df_t2_csv)} samples")
    except Exception as e:
        print(f"✗ Error loading {tf_name}: {e}")

# 9. Offensive-24K-T1 (Offense Detection)
try:
    df_t1 = pd.read_excel(os.path.join(BASE_PATH, "Offensive-24K-T1(Offense Detection).xlsx"))
    df_t1['text'] = df_t1['Tweet'].apply(clean_text)
    # Tag: 0 is Non-Offensive, 1 is Offensive
    df_t1['label'] = df_t1['Tag'].apply(lambda x: 1 if str(x).strip() in ['1', 'OFF'] or x == 1 else 0)
    df_t1 = df_t1.dropna(subset=['text', 'label'])
    all_dfs.append(df_t1[['text', 'label']])
    print(f"✓ Loaded Offensive-24K-T1: {len(df_t1)} samples (Offensive: {sum(df_t1['label']==1)}, Safe: {sum(df_t1['label']==0)})")
except Exception as e:
    print(f"✗ Error loading T1 dataset: {e}")

# 10. Offensive-24K-T2 (Target Identification)
try:
    df_t2 = pd.read_excel(os.path.join(BASE_PATH, "Offensive-24K-T2(Target Identification).xlsx"))
    df_t2['text'] = df_t2['Tweet'].apply(clean_text)
    df_t2['label'] = 1  # All are offensive tweets
    df_t2 = df_t2.dropna(subset=['text', 'label'])
    all_dfs.append(df_t2[['text', 'label']])
    print(f"✓ Loaded Offensive-24K-T2: {len(df_t2)} samples (all offensive)")
except Exception as e:
    print(f"✗ Error loading T2 dataset: {e}")

# 11. Offensive-24K-T3 (Target Type Classification)
try:
    df_t3 = pd.read_excel(os.path.join(BASE_PATH, "Offensive-24K-T3(Target Type Classification).xlsx"))
    df_t3['text'] = df_t3['Tweet'].apply(clean_text)
    df_t3['label'] = 1  # All are offensive tweets
    df_t3 = df_t3.dropna(subset=['text', 'label'])
    all_dfs.append(df_t3[['text', 'label']])
    print(f"✓ Loaded Offensive-24K-T3: {len(df_t3)} samples (all offensive)")
except Exception as e:
    print(f"✗ Error loading T3 dataset: {e}")

# Combine all datasets
df_master = pd.concat(all_dfs, ignore_index=True)
df_master = df_master.dropna(subset=['text', 'label'])
df_master = df_master[df_master['text'].str.len() > 1]

initial_count = len(df_master)
df_master = df_master.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"\n✓ Deduplication: Removed {initial_count - len(df_master)} duplicates.")
print(f"✓ FINAL UNIQUE CORPUS SIZE: {len(df_master)} samples")
print(f"   - Non-Offensive (0): {sum(df_master['label'] == 0)} ({sum(df_master['label'] == 0)/len(df_master)*100:.1f}%)")
print(f"   - Offensive     (1): {sum(df_master['label'] == 1)} ({sum(df_master['label'] == 1)/len(df_master)*100:.1f}%)")

# Save processed dataset
df_master.to_csv(os.path.join(BASE_PATH, "processed_data.csv"), index=False, encoding='utf-8')
print("✓ Saved processed_data.csv")

# ─────────────────────────────────────────────────────────────
# STEP 3: Train / Validation / Test Split (70 / 10 / 20)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 2: Splitting Data Stratified (70% Train, 10% Val, 20% Test)")
print("="*70)

train_val, test_df = train_test_split(df_master, test_size=0.20, random_state=42, stratify=df_master['label'])
train_df, val_df = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['label'])

print(f"Train samples : {len(train_df)}")
print(f"Val samples   : {len(val_df)}")
print(f"Test samples  : {len(test_df)}")

# ─────────────────────────────────────────────────────────────
# STEP 4: TF-IDF Feature Extraction (Context-Aware n-grams 1-3)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 3: TF-IDF Feature Extraction (Context-Aware n-grams 1-3)")
print("="*70)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),      # Unigram, Bigram, and Trigram captures context like "kutta wafadar janwar"
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

print(f"TF-IDF Matrix Shape: {X_train_tfidf.shape}")
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print("✓ Saved models/tfidf_vectorizer.pkl")

# Evaluation Store
evaluation_results = {}

# ─────────────────────────────────────────────────────────────
# STEP 5: Train Machine Learning Models
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 4: Training Machine Learning Classifiers")
print("="*70)

# 1. Linear SVM with Probability Calibration
print("\n--- [1/3] Training Calibrated Linear SVM ---")
base_svm = LinearSVC(C=1.0, max_iter=3000, random_state=42)
# CalibratedClassifierCV equips LinearSVC with predict_proba for accurate confidence scores
svm_calibrated = CalibratedClassifierCV(estimator=base_svm, cv=3)
svm_calibrated.fit(X_train_tfidf, y_train)

y_pred_svm = svm_calibrated.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, zero_division=0)
svm_prec = precision_score(y_test, y_pred_svm, zero_division=0)
svm_rec = recall_score(y_test, y_pred_svm, zero_division=0)

evaluation_results['SVM'] = {
    'test_acc': svm_acc, 'f1': svm_f1, 'precision': svm_prec, 'recall': svm_rec
}
print(f"SVM Test Accuracy: {svm_acc*100:.2f}% | F1: {svm_f1:.4f} | Precision: {svm_prec:.4f} | Recall: {svm_rec:.4f}")
with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
    pickle.dump(svm_calibrated, f)
print("✓ Saved models/svm.pkl (with probability output)")

# 2. Multinomial Naive Bayes
print("\n--- [2/3] Training Multinomial Naive Bayes ---")
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb, zero_division=0)
nb_prec = precision_score(y_test, y_pred_nb, zero_division=0)
nb_rec = recall_score(y_test, y_pred_nb, zero_division=0)

evaluation_results['Naive_Bayes'] = {
    'test_acc': nb_acc, 'f1': nb_f1, 'precision': nb_prec, 'recall': nb_rec
}
print(f"Naive Bayes Test Accuracy: {nb_acc*100:.2f}% | F1: {nb_f1:.4f} | Precision: {nb_prec:.4f} | Recall: {nb_rec:.4f}")
with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(nb, f)
print("✓ Saved models/naive_bayes.pkl")

# 3. Random Forest Classifier
print("\n--- [3/3] Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=120, max_depth=30, min_samples_split=4, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)

y_pred_rf = rf.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)

evaluation_results['Random_Forest'] = {
    'test_acc': rf_acc, 'f1': rf_f1, 'precision': rf_prec, 'recall': rf_rec
}
print(f"Random Forest Test Accuracy: {rf_acc*100:.2f}% | F1: {rf_f1:.4f} | Precision: {rf_prec:.4f} | Recall: {rf_rec:.4f}")
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(rf, f)
print("✓ Saved models/random_forest.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 6: Word Embeddings & Deep Learning (CNN & Bi-LSTM)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" STEP 5: Deep Learning Feature Preparation (Sequences & Embeddings)")
print("="*70)

MAX_LEN = 100
MAX_WORDS = 25000
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 6

# Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text'])

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
print("✓ Saved models/tokenizer.pkl")

def to_seq(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train_seq = to_seq(train_df['text'])
X_val_seq   = to_seq(val_df['text'])
X_test_seq  = to_seq(test_df['text'])

y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat   = to_categorical(y_val,   num_classes=2)
y_test_cat  = to_categorical(y_test,  num_classes=2)

vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)

embedding_matrix = None
if HAS_GENSIM:
    try:
        print("Training Word2Vec embeddings on combined dataset...")
        tokenized_corpus = [text.split() for text in train_df['text']]
        w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4)
        w2v_model.save(os.path.join(MODEL_DIR, "word2vec.model"))
        print("✓ Saved models/word2vec.model")

        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        for word, i in tokenizer.word_index.items():
            if i < vocab_size:
                if word in w2v_model.wv:
                    embedding_matrix[i] = w2v_model.wv[word]
                else:
                    embedding_matrix[i] = np.random.normal(scale=0.4, size=(EMBEDDING_DIM,))
    except Exception as e:
        print(f"Word2Vec notice: {e}")
        embedding_matrix = None

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 4. Deep 1D CNN Architecture
print("\n--- [4/5] Training 1D Convolutional Neural Network (CNN) ---")
if embedding_matrix is not None:
    emb_layer_cnn = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
else:
    emb_layer_cnn = Embedding(vocab_size, EMBEDDING_DIM)

cnn = Sequential([
    emb_layer_cnn,
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
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
cnn_f1 = f1_score(y_test, y_pred_cnn, zero_division=0)
cnn_prec = precision_score(y_test, y_pred_cnn, zero_division=0)
cnn_rec = recall_score(y_test, y_pred_cnn, zero_division=0)

evaluation_results['CNN'] = {
    'test_acc': cnn_acc, 'f1': cnn_f1, 'precision': cnn_prec, 'recall': cnn_rec
}
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}% | F1: {cnn_f1:.4f} | Precision: {cnn_prec:.4f} | Recall: {cnn_rec:.4f}")
cnn.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
print("✓ Saved models/cnn_model.h5")

# 5. Bidirectional LSTM Architecture
print("\n--- [5/5] Training Bidirectional LSTM (Bi-LSTM) ---")
if embedding_matrix is not None:
    emb_layer_lstm = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
else:
    emb_layer_lstm = Embedding(vocab_size, EMBEDDING_DIM)

lstm = Sequential([
    emb_layer_lstm,
    SpatialDropout1D(0.25),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
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
lstm_f1 = f1_score(y_test, y_pred_lstm, zero_division=0)
lstm_prec = precision_score(y_test, y_pred_lstm, zero_division=0)
lstm_rec = recall_score(y_test, y_pred_lstm, zero_division=0)

evaluation_results['LSTM'] = {
    'test_acc': lstm_acc, 'f1': lstm_f1, 'precision': lstm_prec, 'recall': lstm_rec
}
print(f"Bi-LSTM Test Accuracy: {lstm_acc*100:.2f}% | F1: {lstm_f1:.4f} | Precision: {lstm_prec:.4f} | Recall: {lstm_rec:.4f}")
lstm.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
print("✓ Saved models/lstm_model.h5")

# ─────────────────────────────────────────────────────────────
# STEP 7: Comprehensive Results Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  FINAL EVALUATION RESULTS — ALL 5 MODELS (11 DATASETS COMBINED)")
print("="*75)
print(f"  {'Model':<18} {'Test Acc':<14} {'Precision':<14} {'Recall':<14} {'F1-Score':<10}")
print("  " + "-"*70)
for name, m in evaluation_results.items():
    ta = f"{m.get('test_acc', 0)*100:.2f}%"
    pr = f"{m.get('precision', 0)*100:.2f}%"
    rc = f"{m.get('recall', 0)*100:.2f}%"
    f1 = f"{m.get('f1', 0):.4f}"
    print(f"  {name:<18} {ta:<14} {pr:<14} {rc:<14} {f1:<10}")

print("="*75)
print("All 5 Models Trained & Successfully Saved to models/ directory!")
print("="*75)
