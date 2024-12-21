"""
Application that predicts laptop prices based on user input fields.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Create Flask app
app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))


# Define the categorical features and their possible values
categorical_columns = ['Company', 'TypeName', 'Cpu Name', 'OpSys', 'Gpu']
feature_order = ['Company', 'TypeName', 'Inches', 'Ram', 'Gpu', 'OpSys', 'Touchscreen', 'Cpu Name']

    
def preprocess_input(user_input):
    """
    Preprocess form data to match the model's expected input.
    """
    # Convert form data to a DataFrame
    df = pd.DataFrame([user_input])
    
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Ensure all expected columns are present, filling missing ones with 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Align the columns with the model's expected feature order
    df = df[model.feature_names_in_]

    return df.astype(float)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        user_input = {
            "Company": request.form.get("Company"),
            "TypeName": request.form.get("TypeName"),
            "Inches": request.form.get("Inches"),
            "Ram": int(request.form.get("Ram")),
            "gpu": request.form.get("gpu"),
            "os": request.form.get("os"),
            "touchscreen": int(request.form.get("touchscreen")),
            "cpu": int(request.form.get("cpu")),
        }

        # Preprocess the input
        processed_data = preprocess_input(user_input)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Return prediction to the template
        return render_template('index.html', prediction_text=f'Predicted Price: {round(prediction, 2)} Euros')
    
    except Exception as e:
        # Log and handle the error
        print(e)
        return render_template('index.html', prediction_text="An error occurred during prediction. Please try again.")


if __name__ == "__main__":
    app.run(debug=True)
