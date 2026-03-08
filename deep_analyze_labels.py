import pandas as pd
import os

base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"
files = {
    "T1": "Offensive-24K-T1(Offense Detection).xlsx",
    "T3": "Offensive-24K-T3(Target Type Classification).xlsx",
    "Urdu_Abusive": "Dataset of Urdu Abusive Language.xlsx",
    "Roman_Hate": "Hate Speech Roman Urdu (HS-RU-20).xlsx",
    "30k": "final 30,000 dataset_romanurdu.csv",
    "CHate": "CHate.xlsx",
    "GHate": "GHate.xlsx",
    "Cleaned": "cleaned_data.csv"
}

def analyze_file(name, filename):
    path = os.path.join(base_path, filename)
    print(f"\n--- {name} ({filename}) ---")
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path)
            except:
                df = pd.read_csv(path, encoding='latin-1')
        
        print(f"Columns: {df.columns.tolist()}")
        # Identify label columns
        label_cols = [c for c in df.columns if any(x in c.lower() for x in ['tag', 'label', 'target', 'class', 'toxic'])]
        for col in label_cols:
            print(f"Value counts for '{col}':")
            print(df[col].value_counts())
    except Exception as e:
        print(f"Error: {e}")

for name, filename in files.items():
    analyze_file(name, filename)
