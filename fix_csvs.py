import pandas as pd

# Fix heart.csv
heart = pd.read_csv('data/heart.csv')
heart.rename(columns={'num': 'target', 'thalch': 'thalach'}, inplace=True)
heart = heart.drop(columns=['id', 'dataset'], errors='ignore')
heart.to_csv('data/heart.csv', index=False)

# Fix diabetes.csv (just to be safe)
diabetes = pd.read_csv('data/diabetes.csv')
diabetes.to_csv('data/diabetes.csv', index=False)

# Fix breast_cancer.csv
bc = pd.read_csv('data/breast_cancer.csv')
bc = bc.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
bc.to_csv('data/breast_cancer.csv', index=False)

print("All CSV files are fixed ✅")
