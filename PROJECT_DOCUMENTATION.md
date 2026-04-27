# Offense-Guard: Intelligent Offensive Language Detection System
## Comprehensive Project Documentation

---

## 1. Project Overview
Offense-Guard is an AI-powered system designed to detect and mitigate offensive language in Urdu and Roman Urdu. It features a multi-tiered architecture including a Flask-based REST API, a MongoDB-backed storage system, and a cross-platform mobile application.

### 1.1 Objectives
- **Real-time Detection**: Analyze text in <100ms.
- **Bilingual Support**: Handle both Urdu script and Roman Urdu variations.
- **Preventive Intervention**: Implement the 'ReThink' approach to warn users.
- **Continuous Learning**: Use active learning feedback to improve models.

---

## 2. Tools & Technologies

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, REST API |
| **Database** | MongoDB (Users, Feedback, History) |
| **Authentication** | JWT (JSON Web Tokens) |
| **ML Models** | SVM (Primary), Naive Bayes, Random Forest |
| **DL Models** | CNN, LSTM (TensorFlow/Keras) |
| **Features** | TF-IDF, Word2Vec Embeddings |
| **Mobile App** | React Native, Expo, JavaScript |
| **DevOps** | Docker, Docker Compose |

---

## 3. Data & Methodology

### 3.1 Dataset Integration
The system consolidates **11 datasets** with over 110,000 raw samples, processed into 79,564 unique records.
- **Categories**: 
  - 0: Neutral (Safe)
  - 1: Hate Speech
  - 2: Abusive/Profanity
  - 3: Offensive

### 3.2 Preprocessing Pipeline
1. **Cleaning**: URL removal, special character stripping.
2. **Normalization**: Standardizing Roman Urdu (e.g., *kia* → *kya*, *hy* → *hai*).
3. **Tokenization**: NLTK-based word splitting.
4. **Embedding**: Word2Vec for Deep Learning; TF-IDF (5,000 features) for ML.

---

## 4. Model Performance

| Model | Accuracy | F1-Score | Role |
| :--- | :--- | :--- | :--- |
| **SVM (Linear)** | **85.19%** | **0.8341** | **Primary Model** |
| **LSTM** | 83.12% | 0.8150 | Secondary/Experimental |
| **CNN** | 82.45% | 0.8012 | Secondary/Experimental |
| **Naive Bayes** | 80.07% | 0.7869 | Baseline |

---

## 5. System Architecture

### 5.1 Backend API (`app.py`)
- **Endpoints**:
  - `POST /predict`: Unified classification endpoint with model confidence.
  - `POST /api/auth/login/signup`: Secure user management.
  - `GET /stats`: Real-time usage statistics and prediction history.
  - `POST /feedback`: Active learning data collection.

### 5.2 Mobile Application (`mobile_app/`)
- Built with **React Native**, the app provides a seamless chat experience with integrated offensive language warnings.
- **ReThink Logic**: Triggers a modal when a message is classified as category 1, 2, or 3 with high confidence.

---

## 6. Implementation Timeline & Tasks
1. **Initial Research**: Analysis of 11 regional language datasets.
2. **Pipeline Development**: Custom `DataLoader` with multi-class mapping.
3. **ML Training**: Baseline training of NB, SVM, and RF.
4. **DL Development**: Training CNN and LSTM using Word2Vec embeddings.
5. **Backend Engineering**: Flask API with MongoDB and JWT auth.
6. **Mobile Development**: React Native app with real-time API integration.
7. **System Optimization**: Dockerization and performance tuning.

---

## 7. Conclusion
Offense-Guard provides a robust, scalable, and highly accurate solution for regional language moderation. Its combination of traditional ML for speed and Deep Learning for context makes it a state-of-the-art tool for modern digital communication.

---
*Documentation generated after comprehensive codebase analysis.*
