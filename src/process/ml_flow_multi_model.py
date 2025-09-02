import joblib
import os
import numpy as np
import pandas as pd
import urllib.parse
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from flask import Flask, send_from_directory
from flask import Blueprint, request, jsonify
from flask_sqlalchemy import SQLAlchemy

def preprocess_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df.drop("Loan_Status", axis=1), test_df], ignore_index=True)

    # Handle missing values
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
    combined_df["LoanAmount"] = combined_df["LoanAmount"].fillna(combined_df["LoanAmount"].median())

    # Convert 'Dependents' to numerical
    combined_df["Dependents"] = combined_df["Dependents"].replace("3+", 3).astype(int)

    # Label Encoding for binary categorical features
    le = LabelEncoder()
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        combined_df[col] = le.fit_transform(combined_df[col])

    # Separate back into train and test
    X_train_processed = combined_df.iloc[:len(train_df)]
    X_test_processed = combined_df.iloc[len(train_df):]

    # Target variable encoding
    train_df["Loan_Status"] = le.fit_transform(train_df["Loan_Status"])
    y_train = train_df["Loan_Status"]

    # Drop Loan_ID
    X_train_processed = X_train_processed.drop("Loan_ID", axis=1)
    X_test_processed = X_test_processed.drop("Loan_ID", axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    X_train_processed[numerical_cols] = scaler.fit_transform(X_train_processed[numerical_cols])
    X_test_processed[numerical_cols] = scaler.transform(X_test_processed[numerical_cols])

    return X_train_processed, X_test_processed, y_train, scaler



def ml_flow_multi_model():
    # Get the directory of the current Python file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'resources' directory
    resource_filepath = os.path.join(current_file_dir, '..', 'resources')

    X_train, X_test, y_train, scaler = preprocess_data(os.path.join(resource_filepath, 'loan-train.csv'), os.path.join(resource_filepath, 'loan-test.csv'))

    # Handle class imbalance using SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


    print(f"\nOriginal dataset shape: {y_train.value_counts()}")
    print(f"Resampled dataset shape: {y_train_res.value_counts()}")

    # Define models and their hyperparameters for GridSearchCV
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, solver='liblinear'),
            'params': {
                'C': [0.1, 1, 10]
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
        },
        'GradientBoostingClassifier': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }

    best_model = None
    best_score = -1
    best_model_name = ''
    
    # Set MLflow tracking URI (optional, defaults to local ./mlruns)
    # Normalize the resource filepath
    resource_filepath = os.path.normpath(resource_filepath)

    import urllib.parse

    # Construct the tracking URI based on the platform
    if os.name == 'nt':  # Windows
        # Convert Windows path to a valid file:// URI
        # tracking_uri = f"file:///{os.path.abspath(os.path.join(resource_filepath, 'ml_loan_runs')).replace('\\', '/')}"
        # Convert Windows path to a valid file:// URI
        abs_path = os.path.abspath(os.path.join(resource_filepath, 'ml_loan_runs'))
        abs_path = abs_path.replace('\\', '/')  # Perform replacement outside the f-string
        tracking_uri = f"file:///{abs_path}"
    else:  # macOS/Linux
        tracking_uri = f"file://{os.path.abspath(os.path.join(resource_filepath, 'ml_loan_runs'))}"

    # Set the tracking URI
    if not os.getenv("DISABLE_MLFLOW"):
        mlflow.set_tracking_uri(tracking_uri)

    for name, config in models.items():
        if not os.getenv("DISABLE_MLFLOW"):
            # With Mlflow
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_name", name)
                
                print(f"\nTraining and tuning {name}...")
                grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train_res, y_train_res)
                
                best_estimator = grid_search.best_estimator_
                
                print(f"Best parameters for {name}: {grid_search.best_params_}")
                print(f"Best F1-score for {name}: {grid_search.best_score_}")
                
                y_pred = best_estimator.predict(X_train_res)
                accuracy = accuracy_score(y_train_res, y_pred)
                f1 = f1_score(y_train_res, y_pred)
                roc_auc = roc_auc_score(y_train_res, best_estimator.predict_proba(X_train_res)[:, 1])
                
                print(f"Accuracy on resampled training data ({name}): {accuracy}")
                print(f"Classification Report on resampled training data ({name}):\n{classification_report(y_train_res, y_pred)}")
                print(f"ROC AUC Score on resampled training data ({name}): {roc_auc}")

                # Log parameters and metrics to MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("roc_auc_score", roc_auc)
                
                # Infer the model signature
                signature = infer_signature(X_train_res, best_estimator.predict(X_train_res))

                # Define an input example using a sample of the training data
                input_example = X_train_res.iloc[:1]  # Take the first row of the training data as an example

                # Log the model with signature and input_example
                mlflow.sklearn.log_model(sk_model=best_estimator, name="model", input_example=input_example, signature=signature)

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = best_estimator
                    best_model_name = name
        else:
            # Without MLflow
            print(f"\nTraining and tuning {name}...")
            grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train_res, y_train_res)
                
            best_estimator = grid_search.best_estimator_
                
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best F1-score for {name}: {grid_search.best_score_}")
                
            y_pred = best_estimator.predict(X_train_res)
            accuracy = accuracy_score(y_train_res, y_pred)
            f1 = f1_score(y_train_res, y_pred)
            roc_auc = roc_auc_score(y_train_res, best_estimator.predict_proba(X_train_res)[:, 1])
                
            print(f"Accuracy on resampled training data ({name}): {accuracy}")
            print(f"Classification Report on resampled training data ({name}):\n{classification_report(y_train_res, y_pred)}")
            print(f"ROC AUC Score on resampled training data ({name}): {roc_auc}")


            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = best_estimator
                best_model_name = name    
                
                
    print(f"\nBest performing model: {best_model_name} with F1-score: {best_score}")

    # Save the best model and scaler
    joblib.dump(best_model, os.path.join(resource_filepath, 'loan_model.pkl'))
    joblib.dump(scaler, os.path.join(resource_filepath, 'scaler.pkl'))
    print("Best model and scaler saved successfully.")
    

if __name__ == "__main__":
    ml_flow_multi_model()