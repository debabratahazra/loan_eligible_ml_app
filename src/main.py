import joblib
import pandas as pd
import os
import sys
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import mlflow
import mlflow.sklearn

from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, request, jsonify

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.process.pre_process import pre_process
from src.process.train_model import train_model
from src.process.ml_flow_multi_model import ml_flow_multi_model


# Add the parent directory of 'src' to sys.path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

pre_process()
train_model()
ml_flow_multi_model()

from src.api.user_api import user_bp
from src.api.loan_api import loan_bp
from src.database.database import db

# Get the directory of the current Python file
current_file_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(current_file_dir, 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(loan_bp, url_prefix='/api/loan')

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(current_file_dir, 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    print("Starting Loan server on 8090 port number...")
    app.run(host='0.0.0.0', port=8090, debug=True)