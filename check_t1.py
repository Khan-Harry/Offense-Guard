import pandas as pd

# Check T1 label distribution
df = pd.read_excel("d:/Semesters/BSE-6/FYP 2/FYP_Project/Offensive-24K-T1(Offense Detection).xlsx")
print(f"T1 Dataset:")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:")
print(df['Tag'].value_counts())
print(f"\nSample tweets:")
print(df[['Tweet', 'Tag']].head(10))
