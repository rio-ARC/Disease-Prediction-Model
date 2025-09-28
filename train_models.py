import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Dataset paths
datasets = {
    'heart': 'data/heart.csv',
    'diabetes': 'data/diabetes.csv',
    'breast_cancer': 'data/breast_cancer.csv'
}

# Target column names
targets = {
    'heart': 'target',
    'diabetes': 'Outcome',
    'breast_cancer': 'diagnosis'  # 'M'/'B' in UCI dataset
}

for name, path in datasets.items():
    print(f"\n=== Training {name} model ===")
    
    # Load dataset
    df = pd.read_csv(path)
    
    # Convert target to numeric if object
    target_col = targets[name]
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'M': 1, 'B': 0})
    
    # Convert categorical features to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f'models/{name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Print accuracy
    acc = model.score(X_test, y_test)
    print(f"{name} model trained and saved successfully ✅ | Test Accuracy: {acc:.2f}")
