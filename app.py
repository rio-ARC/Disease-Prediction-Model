from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# -------------------- Load Models --------------------
with open('models/heart_model.pkl', 'rb') as f:
    heart_model = pickle.load(f)

with open('models/diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('models/breast_cancer_model.pkl', 'rb') as f:
    bc_model = pickle.load(f)

# -------------------- Feature Lists --------------------
HEART_FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                  'thalach','exang','oldpeak','slope','ca','thal']

DIABETES_FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']
BC_FEATURES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
    'concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
    'concave points_worst','symmetry_worst','fractal_dimension_worst'
]

# -------------------- Input Ranges --------------------
INPUT_RANGES = {
    'heart': {
        'age': (18, 100), 'sex': (0,1), 'cp': (0,3), 'trestbps': (90,200), 'chol': (100,400),
        'fbs': (0,1), 'restecg': (0,2), 'thalach': (60,220), 'exang': (0,1),
        'oldpeak': (0,6), 'slope': (0,2), 'ca': (0,3), 'thal': (1,3)
    },
    'diabetes': {
        'Pregnancies': (0,20), 'Glucose': (50,250), 'BloodPressure': (40,140),
        'SkinThickness': (0,100), 'Insulin': (0,900), 'BMI': (10,70),
        'DiabetesPedigreeFunction': (0,2.5), 'Age': (18,100)
    },
    'breast': {
        'radius_mean': (0,30),'texture_mean': (0,40),'perimeter_mean': (0,200),'area_mean': (0,2500),
        'smoothness_mean': (0,0.2),'compactness_mean': (0,0.5),'concavity_mean': (0,0.5),
        'concave points_mean': (0,0.2),'symmetry_mean': (0,0.3),'fractal_dimension_mean': (0,0.1),
        'radius_se': (0,5),'texture_se': (0,5),'perimeter_se': (0,20),'area_se': (0,500),
        'smoothness_se': (0,0.05),'compactness_se': (0,0.1),'concavity_se': (0,0.2),
        'concave points_se': (0,0.05),'symmetry_se': (0,0.05),'fractal_dimension_se': (0,0.02),
        'radius_worst': (0,40),'texture_worst': (0,50),'perimeter_worst': (0,250),'area_worst': (0,5000),
        'smoothness_worst': (0,0.3),'compactness_worst': (0,1),'concavity_worst': (0,1),
        'concave points_worst': (0,0.3),'symmetry_worst': (0,0.5),'fractal_dimension_worst': (0,0.1)
    }
}

# -------------------- Routes --------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    prediction = None
    message = None

    try:
        # Select model and features based on disease
        if disease == 'heart':
            features = HEART_FEATURES
            model = heart_model
        elif disease == 'diabetes':
            features = DIABETES_FEATURES
            model = diabetes_model
        elif disease == 'breast':
            features = BC_FEATURES
            model = bc_model
        else:
            raise ValueError("Unknown disease")

        # Gather user inputs
        user_input = {}
        for feature in features:
            input_name = feature.replace(' ','_')  # Match HTML input names
            value = request.form.get(input_name)
            if value is None or value.strip() == '':
                raise ValueError(f"{feature} is required")
            value = float(value)

            # Validate range
            min_val, max_val = INPUT_RANGES[disease].get(feature, (None,None))
            if min_val is not None and max_val is not None and not (min_val <= value <= max_val):
                raise ValueError(f"{feature} must be between {min_val} and {max_val}")

            user_input[feature] = value

        # Convert to DataFrame without forcing columns (fixes DiabetesPedigreeFunction issue)
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

    except ValueError as ve:
        message = f"Input Error: {ve}"
    except Exception as e:
        print(e)  # Optional: print error to console for debugging
        message = "Invalid input. Please check all fields."

    return render_template('index.html', prediction=prediction, disease=disease, message=message)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT automatically
    app.run(host="0.0.0.0", port=port, debug=True)

