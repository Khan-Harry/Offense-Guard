import pandas as pd
import os

base_path = "d:/Semesters/BSE-6/FYP 2/FYP_Project"

def analyze_dataset_v2(filename, text_col, label_col, sep=None, header='infer'):
    path = os.path.join(base_path, filename)
    print(f"\n--- {filename} ---")
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            if header == None:
                df = pd.read_csv(path, sep=sep if sep else ',', header=None, names=['text', 'label'])
                label_col = 'label'
                text_col = 'text'
            else:
                df = pd.read_csv(path, sep=sep if sep else ',')
        
        print(f"Total samples: {len(df)}")
        if label_col in df.columns:
            print(f"Label distribution:")
            print(df[label_col].value_counts())
            
            for val in df[label_col].unique():
                print(f"\nExample for Label {val}:")
                subset = df[df[label_col] == val]
                if not subset.empty:
                    # Print 2 examples
                    for i in range(min(2, len(subset))):
                        print(f"- {subset[text_col].iloc[i]}")
        else:
            print(f"Column '{label_col}' not found. Available columns: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"Error: {e}")

analyze_dataset_v2("CHate.xlsx", "Roman Urdu", "RU Original Labels")
analyze_dataset_v2("GHate.xlsx", "Roman Urdu", "RU Original Labels")
analyze_dataset_v2("cleaned_data.csv", "Comment", "Toxic")
analyze_dataset_v2("task_2_train.csv", "text", "label", sep='\t', header=None)
