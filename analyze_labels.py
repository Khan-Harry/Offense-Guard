import pandas as pd
import os

base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"

def analyze_dataset(filename, text_col, label_col, sep=None):
    path = os.path.join(base_path, filename)
    print(f"\n--- {filename} ---")
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, sep=sep if sep else ',')
        
        print(f"Total samples: {len(df)}")
        print(f"Label distribution for '{label_col}':")
        print(df[label_col].value_counts())
        
        # Show some examples for each label to identify "non-offensive"
        for val in df[label_col].unique():
            print(f"\nExample for {val}:")
            print(df[df[label_col] == val][text_col].iloc[0] if not df[df[label_col] == val].empty else "None")
            
    except Exception as e:
        print(f"Error: {e}")

analyze_dataset("CHate.xlsx", "Roman Urdu", "RU Original Labels")
analyze_dataset("GHate.xlsx", "Roman Urdu", "RU Original Labels")
analyze_dataset("cleaned_data.csv", "Comment", "Toxic")
analyze_dataset("task_2_train.csv", "text", "label", sep='\t')
