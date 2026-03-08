import pandas as pd

# Inspect T1, T2, T3 datasets
files = [
    "Offensive-24K-T1(Offense Detection).xlsx",
    "Offensive-24K-T2(Target Identification).xlsx",
    "Offensive-24K-T3(Target Type Classification).xlsx"
]

for f in files:
    print(f"\n{'='*60}")
    print(f"File: {f}")
    print('='*60)
    try:
        df = pd.read_excel(f"d:/Semesters/BSE-6/FYP 2/FYP_Project/{f}", nrows=10)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nData types:")
        print(df.dtypes)
        
        # Check unique values in potential label columns
        for col in df.columns:
            if 'label' in col.lower() or 'target' in col.lower() or 'class' in col.lower():
                print(f"\nUnique values in '{col}': {df[col].unique()}")
    except Exception as e:
        print(f"Error: {e}")
