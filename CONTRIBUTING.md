# Contributing to Offense Guard 🛡️

Welcome! This project is a collaborative FYP. Please follow these guidelines to work efficiently together.

---

## 🔄 Git Workflow (MUST FOLLOW)

### Before Starting Work — Always Pull First!
```bash
git pull origin main
```
> ⚠️ Yeh step skip mat karo — warna conflicts aayenge!

### Daily Workflow
```bash
# 1. Latest changes pull karo
git pull origin main

# 2. Apna kaam karo...

# 3. Changed files stage karo
git add .

# 4. Descriptive commit message likho
git commit -m "feat: add XYZ feature"

# 5. Push karo
git push origin main
```

---

## 📝 Commit Message Format

Use this format for clear history:

| Prefix | Use For |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `data:` | Dataset or model changes |
| `docs:` | Documentation update |
| `refactor:` | Code cleanup |
| `test:` | Tests |

**Examples:**
```
feat: add T3 dataset loading to data_loader
fix: resolve Unicode error in retrain script
data: retrain all models on 11 datasets
docs: update README with run instructions
```

---

## 📁 Project Structure

```
Offense-Guard/
├── app.py                    # Flask Backend API (main server)
├── data_loader.py            # Dataset loading & preprocessing
├── feature_extraction.py     # TF-IDF & Word2Vec features
├── train_ml_models.py        # ML model training (SVM, NB, RF)
├── train_dl_models.py        # DL model training (CNN, LSTM)
├── retrain_all_models.py     # Retrain ALL 5 models at once
├── retrain_models.py         # Incremental retraining from feedback
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from git
├── CONTRIBUTING.md           # This file
├── README.md                 # Project overview
│
├── models/                   # Trained models (NOT in git — generate locally)
│   ├── svm.pkl              # Best model
│   ├── tfidf_vectorizer.pkl
│   └── ...
│
├── mobile_app/               # React Native mobile app
│   ├── App.js
│   ├── src/
│   └── package.json
│
└── results/                  # Evaluation results & plots
```

---

## ⚙️ First-Time Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Khan-Harry/Offense-Guard.git
cd Offense-Guard

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate ML models locally (not stored in git)
python retrain_all_models.py

# 4. Install mobile app dependencies
cd mobile_app
npm install
cd ..

# 5. Run Flask backend
python app.py
```

---

## ⚠️ Important Rules

1. **Never commit model `.pkl` / `.h5` files** — they are in `.gitignore`. Run `retrain_all_models.py` locally.
2. **Never commit `processed_data.csv`** — it's large and auto-generated.
3. **Always `git pull` before starting work.**
4. **Do NOT push directly if you know the other person is also making changes** — coordinate first.
5. **If there's a merge conflict**, discuss before resolving.

---

## 🧑‍💻 Team

| Name | Role | GitHub |
|------|------|--------|
| Khan Harry | ML + Backend | [@Khan-Harry](https://github.com/Khan-Harry) |
| Collaborator | Mobile App / Other | — |

---

## 📞 Conflict Resolution

Agar merge conflict aaye:
```bash
git pull origin main
# Conflicts fix karo manually in VS Code
git add .
git commit -m "fix: resolve merge conflict"
git push origin main
```
