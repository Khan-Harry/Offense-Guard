# Quick Reference Guide

## 🚀 Running the Application

### Option 1: Run Existing Trained Model
```bash
cd "d:/Semesters/BSE-6/FYP 2/FYP_Project"
python app.py
# Open browser: http://localhost:5000
```

### Option 2: Retrain Models from Scratch
```bash
# Step 1: Preprocess data
python data_loader.py

# Step 2: Train models
python train_ml_models.py

# Step 3: Run application
python app.py
```

## 📁 Important Files

### Core Implementation
- `app.py` - Flask backend (START HERE)
- `data_loader.py` - Data preprocessing
- `train_ml_models.py` - Model training
- `templates/index.html` - Frontend UI
- `static/style.css` - Styling
- `static/script.js` - Frontend logic

### Documentation
- `PROJECT_DOCUMENTATION.md` - Full 30+ page documentation
- `README.md` - Quick start guide
- `walkthrough.md` - Project walkthrough

### Models (Already Trained)
- `models/svm.pkl` - Best model (88.7% accuracy)
- `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `models/naive_bayes.pkl` - Naive Bayes
- `models/random_forest.pkl` - Random Forest

## 📊 Key Results

### Model Performance
- **Best Model**: SVM
- **Accuracy**: 88.7%
- **Precision**: 88.65%
- **Recall**: 85.38%
- **F1-Score**: 86.98%

### Dataset
- **Total Samples**: 41,845 (after deduplication)
- **Offensive**: 24,516 (58.6%)
- **Non-Offensive**: 17,329 (41.4%)

## 🎯 Features Implemented

✅ Data preprocessing for Urdu & Roman Urdu
✅ TF-IDF feature extraction
✅ 3 ML models (NB, SVM, RF)
✅ Flask REST API
✅ Modern web interface
✅ ReThink warning system
✅ Feedback collection
✅ Real-time classification (<100ms)
✅ Comprehensive documentation

## 🔧 API Endpoints

### POST /predict
```json
Request:  {"text": "your message"}
Response: {
  "prediction": "offensive",
  "confidence": 0.85,
  "should_warn": true
}
```

### POST /feedback
```json
Request: {
  "text": "message",
  "predicted_label": "offensive",
  "actual_label": "non-offensive",
  "user_action": "posted"
}
```

### GET /stats
```json
Response: {
  "total_feedback": 150,
  "accuracy_from_feedback": 0.89
}
```

## 🎓 For FYP Presentation

### Key Points to Highlight
1. **Problem**: Lack of offensive language detection for Urdu
2. **Solution**: AI-powered system with 88.7% accuracy
3. **Innovation**: ReThink preventive warnings
4. **Dataset**: 40,000+ samples
5. **Technology**: SVM with TF-IDF
6. **Deployment**: Working web application

### Demo Flow
1. Show main interface
2. Type offensive Urdu text
3. Show ReThink warning modal
4. Type safe text
5. Show safe result
6. Explain feedback loop
7. Show system statistics

## 📝 Project Structure
```
FYP_Project/
├── app.py                      # Flask backend ⭐
├── data_loader.py              # Data preprocessing
├── train_ml_models.py          # Model training
├── feature_extraction.py       # TF-IDF
├── models/                     # Trained models ⭐
│   ├── svm.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html             # Frontend ⭐
├── static/
│   ├── style.css
│   └── script.js
├── PROJECT_DOCUMENTATION.md    # Full docs ⭐
├── README.md                   # Quick start ⭐
└── processed_data.csv          # Cleaned data
```

## 🐛 Troubleshooting

### Flask app won't start
```bash
# Check if models exist
ls models/

# If missing, train models first
python train_ml_models.py
```

### Import errors
```bash
pip install pandas numpy scikit-learn nltk flask openpyxl
```

### Port already in use
```python
# In app.py, change port:
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📞 Support

For questions or issues:
1. Check PROJECT_DOCUMENTATION.md
2. Check README.md
3. Review code comments
4. Contact supervisor

---

**Project Status**: ✅ COMPLETE
**Ready for**: Deployment, Presentation, Evaluation
