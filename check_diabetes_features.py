import pickle

# Load the trained diabetes model
with open('models/diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

# Check which feature names the model expects
try:
    print("Expected feature names by the model:")
    print(diabetes_model.feature_names_in_)
except AttributeError:
    print("This model does not have 'feature_names_in_'. It might expect input as plain arrays.")
