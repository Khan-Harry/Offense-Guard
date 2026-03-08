import pandas as pd
import os

# Inspect specific remaining datasets with more detail
base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"

def inspect_file(filename, **kwargs):
    path = os.path.join(base_path, filename)
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print('='*60)
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(path, nrows=5)
        else:
            df = pd.read_csv(path, nrows=5, **kwargs)
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 2 rows:")
        print(df.head(2))
        
        # Check for unique values in label columns
        for col in df.columns:
            if any(word in col.lower() for word in ['label', 'target', 'class', 'tag', 'toxic']):
                print(f"Unique values in '{col}': {df[col].unique()}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

inspect_file("CHate.xlsx")
inspect_file("GHate.xlsx")
inspect_file("task_2_train.csv", sep='\t', header=None, names=['text', 'label'])
inspect_file("task_2_test.csv", sep='\t', header=None, names=['text', 'label'])
