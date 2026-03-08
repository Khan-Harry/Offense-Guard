
import pandas as pd
import os

files = [
    "Hate Speech Roman Urdu (HS-RU-20).xlsx",
    "Dataset of Urdu Abusive Language.xlsx",
    "final 30,000 dataset_romanurdu.csv"
]

base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"

for f in files:
    path = os.path.join(base_path, f)
    print(f"--- {f} ---")
    try:
        if f.endswith('.xlsx'):
            df = pd.read_excel(path, nrows=5)
        else:
            df = pd.read_csv(path, nrows=5)
        print(df.columns)
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {f}: {e}")
    print("\n")
