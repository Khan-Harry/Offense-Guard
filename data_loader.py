"""
Data Loader and Preprocessing Module
Handles loading, cleaning, and preprocessing of Urdu and Roman Urdu datasets
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, base_path="d:/Semesters/BSE-6/FYP 2/FYP_Project"):
        self.base_path = base_path
        self.data = None
        
    def load_datasets(self):
        """Load and merge all available datasets"""
        all_data = []
        
        # Load Roman Urdu Hate Speech dataset
        try:
            df1 = pd.read_excel(os.path.join(self.base_path, "Hate Speech Roman Urdu (HS-RU-20).xlsx"))
            # Map H (Hostile) to 1 (Hate Speech), N (Neutral) to 0 (Neutral)
            df1['label'] = df1['Neutral (N) / Hostile (H)'].apply(lambda x: 1 if x == 'H' else 0)
            df1['text'] = df1['Sentence']
            all_data.append(df1[['text', 'label']])
            print(f"✓ Loaded Roman Urdu Hate Speech: {len(df1)} samples")
        except Exception as e:
            print(f"✗ Error loading Roman Urdu dataset: {e}")
        
        # Load Urdu Abusive Language dataset
        try:
            df2 = pd.read_excel(os.path.join(self.base_path, "Dataset of Urdu Abusive Language.xlsx"))
            # target 1.0 is abusive (2), 0.0 is neutral (0)
            df2['label'] = df2['target'].apply(lambda x: 2 if x == 1 else 0)
            df2['text'] = df2['no stop']
            all_data.append(df2[['text', 'label']])
            print(f"✓ Loaded Urdu Abusive Language: {len(df2)} samples")
        except Exception as e:
            print(f"✗ Error loading Urdu dataset: {e}")
        
        # Load Roman Urdu 30k dataset
        try:
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df3 = pd.read_csv(os.path.join(self.base_path, "final 30,000 dataset_romanurdu.csv"), encoding=encoding)
                    # Map H (Hostile) to 1 (Hate Speech), N (Neutral) to 0
                    df3['label'] = df3['label'].apply(lambda x: 1 if x == 'H' else 0)
                    df3['text'] = df3['tweets']
                    df3 = df3.dropna(subset=['text', 'label'])
                    all_data.append(df3[['text', 'label']])
                    print(f"✓ Loaded Roman Urdu 30k dataset: {len(df3)} samples")
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"✗ Error loading Roman Urdu 30k dataset: {e}")
        
        # Load CHate.xlsx (Conversational Hate)
        try:
            df_chate = pd.read_excel(os.path.join(self.base_path, "CHate.xlsx"))
            # H -> 1 (Hate Speech), O -> 3 (Offensive)
            df_chate['label'] = df_chate['RU Original Labels'].apply(lambda x: 1 if x == 'H' else 3)
            df_chate['text'] = df_chate['Roman Urdu']
            df_chate = df_chate.dropna(subset=['text', 'label'])
            all_data.append(df_chate[['text', 'label']])
            print(f"✓ Loaded CHate: {len(df_chate)} samples")
        except Exception as e:
            print(f"✗ Error loading CHate: {e}")

        # Load GHate.xlsx (Generalized Hate)
        try:
            df_ghate = pd.read_excel(os.path.join(self.base_path, "GHate.xlsx"))
            # H -> 1 (Hate Speech), O -> 3 (Offensive)
            df_ghate['label'] = df_ghate['RU Original Labels'].apply(lambda x: 1 if x == 'H' else 3)
            df_ghate['text'] = df_ghate['Roman Urdu']
            df_ghate = df_ghate.dropna(subset=['text', 'label'])
            all_data.append(df_ghate[['text', 'label']])
            print(f"✓ Loaded GHate: {len(df_ghate)} samples")
        except Exception as e:
            print(f"✗ Error loading GHate: {e}")

        # Load cleaned_data.csv
        try:
            df_cleaned = pd.read_csv(os.path.join(self.base_path, "cleaned_data.csv"))
            # 1 -> 0 (Neutral), 2 -> 1 (Hate Speech), 0 -> 2 (Abusive), 3,4 -> 3 (Offensive)
            def map_cleaned(x):
                if x == 1: return 0
                if x == 2: return 1
                if x == 0: return 2
                return 3
            df_cleaned['label'] = df_cleaned['Toxic'].apply(map_cleaned)
            df_cleaned['text'] = df_cleaned['Comment']
            df_cleaned = df_cleaned.dropna(subset=['text', 'label'])
            all_data.append(df_cleaned[['text', 'label']])
            print(f"✓ Loaded cleaned_data: {len(df_cleaned)} samples")
        except Exception as e:
            print(f"✗ Error loading cleaned_data: {e}")

        # Load task_2 datasets
        for task_file in ["task_2_train.csv", "task_2_test.csv"]:
            try:
                df_task = pd.read_csv(os.path.join(self.base_path, task_file), sep='\t', header=None, names=['text', 'label'])
                # Same mapping as cleaned_data
                df_task['label'] = df_task['label'].apply(map_cleaned)
                df_task = df_task.dropna(subset=['text', 'label'])
                all_data.append(df_task[['text', 'label']])
                print(f"✓ Loaded {task_file}: {len(df_task)} samples")
            except Exception as e:
                print(f"✗ Error loading {task_file}: {e}")
        
        # Merge all datasets
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            
            # Remove any NaN values
            self.data = self.data.dropna(subset=['text', 'label'])
            
            print(f"\n✓ Total samples loaded: {len(self.data)}")
            print(f"  - Neutral (0): {sum(self.data['label'] == 0)}")
            print(f"  - Hate Speech (1): {sum(self.data['label'] == 1)}")
            print(f"  - Abusive (2): {sum(self.data['label'] == 2)}")
            print(f"  - Offensive (3): {sum(self.data['label'] == 3)}")
        else:
            raise ValueError("No datasets could be loaded!")
        
        return self.data
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Roman Urdu variations (common spelling variations)
        # Example: "kya" vs "kia", "hai" vs "hy"
        text = text.replace('kia', 'kya')
        text = text.replace('hy', 'hai')
        text = text.replace('hain', 'han')
        
        return text.lower()
    
    def preprocess(self):
        """Apply preprocessing to the dataset"""
        if self.data is None:
            raise ValueError("No data loaded! Call load_datasets() first.")
        
        # Clean text
        self.data['text'] = self.data['text'].apply(self.clean_text)
        
        # Remove empty texts
        self.data = self.data[self.data['text'].str.len() > 0]
        
        # Remove duplicates
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['text'])
        print(f"\n✓ Removed {initial_count - len(self.data)} duplicates")
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        print(f"✓ Preprocessing complete. Final dataset size: {len(self.data)}")
        
        return self.data
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        if self.data is None:
            raise ValueError("No data available! Call load_datasets() and preprocess() first.")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.data['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val['label']
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  - Train: {len(train)} samples")
        print(f"  - Validation: {len(val)} samples")
        print(f"  - Test: {len(test)} samples")
        
        return train, val, test
    
    def save_processed_data(self, output_dir=None):
        """Save processed data to CSV"""
        if output_dir is None:
            output_dir = self.base_path
        
        output_path = os.path.join(output_dir, "processed_data.csv")
        self.data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Processed data saved to: {output_path}")
        
        return output_path


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load datasets
    data = loader.load_datasets()
    
    # Preprocess
    data = loader.preprocess()
    
    # Split data
    train, val, test = loader.split_data()
    
    # Save processed data
    loader.save_processed_data()
    
    # Display sample
    print("\n--- Sample Data ---")
    print(data.head(10))
