# Loan Eligibility Prediction API

A RESTful API application built with Flask that predicts loan eligibility using machine learning.

## Features

- Machine learning model trained on loan data
- RESTful API endpoints for predictions
- Web interface for easy testing
- Cross-platform compatibility

## API Endpoints

### Health Check
- **GET** `/api/loan/health`
- Returns the health status of the API

### Loan Prediction
- **POST** `/api/loan/predict`
- Predicts loan eligibility based on applicant data

#### Request Body Example:
```json
{
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
```

#### Response Example:
```json
{
  "prediction": "Approved",
  "confidence": 0.85,
  "probability_approved": 0.85,
  "probability_rejected": 0.15
}
```

## Required Input Fields

- **Gender**: "Male" or "Female"
- **Married**: "Yes" or "No"
- **Dependents**: "0", "1", "2", or "3+"
- **Education**: "Graduate" or "Not Graduate"
- **Self_Employed**: "Yes" or "No"
- **ApplicantIncome**: Numeric value (applicant's income)
- **CoapplicantIncome**: Numeric value (co-applicant's income, can be 0)
- **LoanAmount**: Numeric value (loan amount requested)
- **Loan_Amount_Term**: Numeric value (loan term in months, typically 360)
- **Credit_History**: 1 (good credit history) or 0 (poor credit history)
- **Property_Area**: "Urban", "Semiurban", or "Rural"

## Installation and Setup

### Prerequisites
- Python 3.11+
- pip

### Local Development

1. Clone or extract the project files
2. Navigate to the project directory:
   ```bash
   cd loan_eligible_ml_app
   ```

3. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python src/main.py
   ```

6. Access the application:
   - Web Interface: http://localhost:8090
   - API Base URL: http://localhost:8090/api/loan

## Project Structure

```
loan_api/
├── src/
│   ├── api/          # API route handlers
│   │   ├── loan_api.py      # Loan prediction endpoints
│   │   └── user_api.py      # User management endpoints
│   ├── static/          # Frontend files
│   │   └── index.html   # Web interface
│   ├── database/        # Database files
│   ├── main.py          # Application entry point
│   ├── loan_eligibility_model.pkl  # Trained ML model
│   └── scaler.pkl       # Feature scaler
├── venv/                # Virtual environment
├── requirements.txt     # Python dependencies
├── tests/               # Unit tests
│   └── test_api.py      # API unit tests
└── README.md           # This file
```

## Model Information

The machine learning model is trained using:
- **Algorithm**: RandomForestClassifier (selected as best performing after hyperparameter tuning)
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes.
- **Training F1-score (on resampled data)**: 0.8279 (RandomForestClassifier)
- **Training Accuracy (on resampled data)**: 1.0 (RandomForestClassifier) - *Note: This high accuracy on training data might indicate overfitting. Further validation on unseen data is recommended.*
- **Features**: 11 input features including demographic and financial data
- **Preprocessing**: Categorical encoding, missing value imputation, and feature scaling

## Deployment

The application is configured to run on `0.0.0.0:8090` and supports CORS for cross-origin requests, making it suitable for deployment on cloud platforms.

### Production Considerations

1. Use a production WSGI server (e.g., Gunicorn)
2. Set up proper environment variables for configuration
3. Implement proper logging and monitoring
4. Use a production database instead of SQLite
5. Set up SSL/TLS for HTTPS

## Testing

You can test the API using:

1. **Web Interface**: Navigate to the root URL and use the form
2. **cURL**: 
   ```bash
   curl -X POST -H "Content-Type: application/json" \
   -d '{"Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate","Self_Employed":"No","ApplicantIncome":5000,"CoapplicantIncome":2000,"LoanAmount":150,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Urban"}' \
   http://localhost:8090/api/loan/predict
   ```
3. **Postman**: Import the API endpoints and test with sample data

### Running Unit Tests

1.  **Navigate to the project directory**:
    ```bash
    cd loan_eligible_ml_app
    ```

2.  **Activate the virtual environment**:
    ```bash
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install `pytest` and `pytest-flask`**:
    ```bash
    pip install pytest pytest-flask
    ```

4.  **Run the tests**:
   In Windows:
   ```bash
      set PYTHONPATH=%CD%
      pytest
   ```
   In Linux:

   ```bash
   export PYTHONPATH=$PYTHONPATH:/home/loan_eligible_ml_app && pytest
   ```

## License

This project is for educational and demonstration purposes.

