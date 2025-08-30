import joblib

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import mlflow
import mlflow.sklearn

from flask import Blueprint, request, jsonify
import os
import joblib
import numpy as np

from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy

loan_bp = Blueprint('loan', __name__)

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler
model_path = os.path.join(current_file_dir, '..', 'resources','loan_model.pkl')
scaler_path = os.path.join(current_file_dir, '..', 'resources','scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@loan_bp.route('/predict', methods=['POST'])
def predict_loan_eligibility():
    try:
        data = request.get_json()
        
        # Expected input features
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Preprocess the input data
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict([input_data])[0]
        prediction_proba = model.predict_proba([input_data])[0]
        
        # Convert prediction to human-readable format
        result = 'Approved' if prediction == 1 else 'Rejected'
        confidence = max(prediction_proba)
        
        return jsonify({
            'prediction': result,
            'confidence': float(confidence),
            'probability_approved': float(prediction_proba[1]),
            'probability_rejected': float(prediction_proba[0])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_input(data):
    # Convert categorical variables to numerical
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 0, 'Not Graduate': 1}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    
    # Handle dependents
    dependents = data['Dependents']
    if dependents == '3+':
        dependents = 3
    else:
        dependents = int(dependents)
    
    # Create feature array
    features = [
        gender_map.get(data['Gender'], 1),
        married_map.get(data['Married'], 1),
        dependents,
        education_map.get(data['Education'], 0),
        self_employed_map.get(data['Self_Employed'], 0),
        float(data['ApplicantIncome']),
        float(data['CoapplicantIncome']),
        float(data['LoanAmount']),
        float(data['Loan_Amount_Term']),
        float(data['Credit_History']),
        property_area_map.get(data['Property_Area'], 2)
    ]
    
    # Scale numerical features
    numerical_indices = [5, 6, 7, 8]  # ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
    features_array = np.array(features).reshape(1, -1)
    features_array[:, numerical_indices] = scaler.transform(features_array[:, numerical_indices])
    
    return features_array[0]

@loan_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Loan prediction API is running'})

