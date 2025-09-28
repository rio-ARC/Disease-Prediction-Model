import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model paths and dataset paths
models = {
    'Heart Disease': 'models/heart_model.pkl',
    'Diabetes': 'models/diabetes_model.pkl',
    'Breast Cancer': 'models/breast_cancer_model.pkl'
}

datasets = {
    'Heart Disease': 'data/heart.csv',
    'Diabetes': 'data/diabetes.csv',
    'Breast Cancer': 'data/breast_cancer.csv'
}

targets = {
    'Heart Disease': 'target',
    'Diabetes': 'Outcome',
    'Breast Cancer': 'diagnosis'
}

for disease in models.keys():
    print(f"\n=== Evaluating {disease} Model ===")
    
    # Load model
    if not os.path.exists(models[disease]):
        print(f"❌ Model file not found: {models[disease]}")
        continue
    
    try:
        with open(models[disease], 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        continue
    
    # Load dataset
    df = pd.read_csv(datasets[disease])
    
    # Convert target to numeric if needed
    target_col = targets[disease]
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'M': 1, 'B': 0})
    
    # Convert categorical features to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Features and target
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    
    # Predict
    y_pred = model.predict(X)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print results
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")
