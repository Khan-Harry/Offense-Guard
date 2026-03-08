# AI-Powered Offensive Language Detection System
### Real-Time Detection for Urdu & Roman Urdu

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.7%25-success.svg)]()

## 🎯 Overview

An AI-powered system that detects offensive language in Urdu and Roman Urdu text in real-time, provides preventive "ReThink" warnings, and continuously improves through user feedback.

### ✨ Key Features

- 🤖 **AI-Powered**: SVM classifier trained on 40,000+ samples
- ⚡ **Real-Time**: <100ms prediction time
- 🌐 **Bilingual**: Supports Urdu script and Roman Urdu
- 🛡️ **ReThink Warnings**: Preventive intervention before posting
- 📊 **High Accuracy**: 88.7% test accuracy
- 🔄 **Continuous Learning**: Feedback loop for improvement
- 💻 **Web Interface**: Modern, user-friendly UI

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **88.7%** | **88.65%** | **85.38%** | **86.98%** |
| Naïve Bayes | 84.35% | 82.57% | 86.96% | 84.71% |
| Random Forest | 80.29% | 91.07% | 67.02% | 77.22% |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone/Download the project**
```bash
cd "d:/Semesters/BSE-6/FYP 2/FYP_Project"
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn nltk flask openpyxl
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
```
http://localhost:5000
```

That's it! 🎉

## 📁 Project Structure

```
FYP_Project/
├── app.py                      # Flask backend API
├── data_loader.py              # Data preprocessing
├── train_ml_models.py          # Model training
├── feature_extraction.py       # TF-IDF implementation
├── requirements.txt            # Dependencies
├── PROJECT_DOCUMENTATION.md    # Full documentation
├── models/                     # Trained models
│   ├── svm.pkl                # Best model (SVM)
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   └── ...
├── templates/
│   └── index.html             # Frontend UI
└── static/
    ├── style.css              # Styling
    └── script.js              # Frontend logic
```

## 🎨 Screenshots

### Main Interface
![Main Interface](screenshots/main_interface.png)

### ReThink Warning
![ReThink Warning](screenshots/rethink_warning.png)

### Results Display
![Results](screenshots/results.png)

## 🔧 Usage

### Web Interface

1. **Type your message** in Urdu or Roman Urdu
2. **Click "Check Message"**
3. **View results**:
   - ✅ Safe: Message is non-offensive
   - ⚠️ Warning: ReThink modal appears for offensive content
4. **Choose action**:
   - Edit Message
   - Post Anyway
   - Cancel

### API Usage

#### Predict Endpoint
```python
import requests

response = requests.post('http://localhost:5000/predict', 
    json={'text': 'your message here'})
print(response.json())
```

**Response**:
```json
{
  "text": "your message here",
  "prediction": "offensive",
  "confidence": 0.85,
  "should_warn": true,
  "timestamp": "2026-01-22T22:00:00"
}
```

#### Feedback Endpoint
```python
requests.post('http://localhost:5000/feedback', json={
    'text': 'the message',
    'predicted_label': 'offensive',
    'actual_label': 'non-offensive',
    'user_action': 'posted'
})
```

## 📚 Documentation

For complete documentation, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

Topics covered:
- Introduction & Problem Statement
- Literature Review
- Methodology & System Architecture
- Implementation Details
- Results & Evaluation
- Scope & Limitations
- Future Enhancements

## 🧪 Training Your Own Model

```bash
# 1. Load and preprocess data
python data_loader.py

# 2. Train models
python train_ml_models.py

# 3. Models will be saved in models/ directory
```

## 📊 Dataset

- **Total Samples**: 47,082 (41,845 after deduplication)
- **Offensive**: 24,516 (58.6%)
- **Non-Offensive**: 17,329 (41.4%)
- **Sources**:
  - Hate Speech Roman Urdu (HS-RU-20): 5,000
  - Dataset of Urdu Abusive Language: 12,083
  - Roman Urdu 30K: 29,999

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **ML/NLP**: scikit-learn, NLTK
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: pandas, numpy

## 🔮 Future Enhancements

- [ ] Deep Learning models (CNN, LSTM, Transformers)
- [ ] Mobile application (Android/iOS)
- [ ] Browser extension
- [ ] Multi-class classification (hate speech, profanity, etc.)
- [ ] Explainability (highlight offensive words)
- [ ] Multilingual support (Punjabi, Pashto, Sindhi)

## 🤝 Contributing

This is a Final Year Project. For suggestions or improvements, please contact the author.

## 📝 License

This project is developed as part of academic research.

## 👨‍💻 Author

**Final Year Project**
- Institution: [Your University]
- Year: 2026
- Supervisor: [Supervisor Name]

## 📧 Contact

For questions or feedback:
- Email: [Your Email]
- GitHub: [Your GitHub]

## 🙏 Acknowledgments

- Dataset providers
- scikit-learn community
- Flask framework
- ReThink project inspiration

---

**⭐ If you find this project useful, please give it a star!**

---

*Developed with ❤️ for promoting civil online discourse in Urdu*
