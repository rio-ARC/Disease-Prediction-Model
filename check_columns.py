import pandas as pd

# List of CSV files
files = ['data/heart.csv', 'data/diabetes.csv', 'data/breast_cancer.csv']

for file in files:
    df = pd.read_csv(file)
    print(f"\nColumns in {file}:")
    print(df.columns.tolist())
