import pandas as pd

# Inspect the Roman Urdu 30k dataset more carefully
for encoding in ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']:
    try:
        df = pd.read_csv("d:/Semesters/BSE-6/FYP 2/FYP_Project/final 30,000 dataset_romanurdu.csv", 
                        encoding=encoding, nrows=10)
        print(f"\n=== Encoding: {encoding} ===")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nData types:")
        print(df.dtypes)
        break
    except Exception as e:
        print(f"Failed with {encoding}: {e}")
