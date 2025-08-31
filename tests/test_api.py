import pytest
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.main import app
from src.api.loan_api import preprocess_input
import numpy as np

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.mark.skipif("mlflow" in globals(), reason="Skipping mlflow tests")
def test_health_endpoint(client):
    """Test the /api/loan/health endpoint."""
    response = client.get("/api/loan/health")
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "healthy", "message": "Loan prediction API is running"}

@pytest.mark.skipif("mlflow" in globals(), reason="Skipping mlflow tests")
def test_predict_valid_input(client):
    """Test the /api/loan/predict endpoint with valid input."""
    valid_data = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    response = client.post("/api/loan/predict", data=json.dumps(valid_data), content_type="application/json")
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert "prediction" in response_data
    assert "confidence" in response_data
    assert "probability_approved" in response_data
    assert "probability_rejected" in response_data

@pytest.mark.skipif("mlflow" in globals(), reason="Skipping mlflow tests")
def test_predict_missing_field(client):
    """Test the /api/loan/predict endpoint with a missing field."""
    invalid_data = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1
        # Missing "Property_Area"
    }
    response = client.post("/api/loan/predict", data=json.dumps(invalid_data), content_type="application/json")
    assert response.status_code == 400
    assert json.loads(response.data) == {"error": "Missing field: Property_Area"}

@pytest.mark.skipif("mlflow" in globals(), reason="Skipping mlflow tests")
def test_preprocess_input():
    """Test the preprocess_input function."""
    sample_data = {
        "Gender": "Female",
        "Married": "No",
        "Dependents": "3+",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 6000,
        "CoapplicantIncome": 1000,
        "LoanAmount": 200,
        "Loan_Amount_Term": 180,
        "Credit_History": 0,
        "Property_Area": "Rural"
    }
    # This test assumes that the scaler and model are loaded correctly within the loan.py route
    # and that the preprocess_input function uses them. For a true unit test, you might mock them.
    # However, for integration with the existing setup, we'll test its output format.
    
    # The actual values after scaling depend on the scaler.pkl, so we can only check the shape and type.
    processed_output = preprocess_input(sample_data)
    assert isinstance(processed_output, np.ndarray)
    assert processed_output.shape == (11,)
    # Further checks could involve comparing with expected scaled values if the scaler was fixed.



