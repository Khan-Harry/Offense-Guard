import pandas as pd
import os

# Inspect remaining datasets
files = [
    "CHate.xlsx",
    "GHate.xlsx",
    "cleaned_data.csv",
    "task_2_test.csv",
    "task_2_train.csv"
]

base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"

for f in files:
    path = os.path.join(base_path, f)
    print(f"\n{'='*60}")
    print(f"File: {f}")
    print('='*60)
    try:
        if f.endswith('.xlsx'):
            df = pd.read_excel(path, nrows=10)
        else:
            df = pd.read_csv(path, nrows=10)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Check for label-like columns
        potential_labels = [c for c in df.columns if any(word in c.lower() for word in ['label', 'target', 'class', 'tag', 'category', 'hostile'])]
        if potential_labels:
            print(f"\nPotential labels found: {potential_labels}")
            for col in potential_labels:
                print(f"Unique values in '{col}': {df[col].unique()}")
        
    except Exception as e:
        print(f"Error reading {f}: {e}")
