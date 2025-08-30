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

def preprocess_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df.drop("Loan_Status", axis=1), test_df], ignore_index=True)

    # Handle missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
    combined_df['LoanAmount'] = combined_df['LoanAmount'].fillna(combined_df['LoanAmount'].median())

    # Convert 'Dependents' to numerical
    combined_df['Dependents'] = combined_df['Dependents'].replace('3+', 3).astype(int)

    # Label Encoding for binary categorical features
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        combined_df[col] = le.fit_transform(combined_df[col])

    # Separate back into train and test
    X_train_processed = combined_df.iloc[:len(train_df)]
    X_test_processed = combined_df.iloc[len(train_df):]

    # Target variable encoding
    train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])
    y_train = train_df['Loan_Status']

    # Drop Loan_ID
    X_train_processed = X_train_processed.drop('Loan_ID', axis=1)
    X_test_processed = X_test_processed.drop('Loan_ID', axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    X_train_processed[numerical_cols] = scaler.fit_transform(X_train_processed[numerical_cols])
    X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])

    return X_train_processed, X_test_processed, y_train


    
def pre_process():
    # Get the directory of the current Python file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'resources' directory
    csv_filepath = os.path.join(current_file_dir, '..', 'resources')

    X_train, X_test, y_train = preprocess_data(os.path.join(csv_filepath, 'loan-train.csv'), os.path.join(csv_filepath, 'loan-test.csv'))

    print("\n--- Preprocessed X_train Info ---")
    print(X_train.info())

    print("\n--- Preprocessed X_train Head ---")
    print(X_train.head())

    print("\n--- Preprocessed X_test Info ---")
    print(X_test.info())

    print("\n--- Preprocessed X_test Head ---")
    print(X_test.head())

if __name__ == "__main__":
    pre_process()