import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from xgboost import XGBClassifier

ENCODER_PATH = "models/encoders/"
MODEL_PATH = "models/fraud_detection_model.pkl"

# Load saved encoders
def load_encoder(file_name):
    with open(os.path.join(ENCODER_PATH, file_name), 'rb') as f:
        return pickle.load(f)

def preprocess_custom_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Timestamp → Unix
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time']).astype(int) / 10**9

    # DOB → Age
    df['dob'] = pd.to_datetime(df['dob'])
    df['dob'] = (pd.Timestamp.now() - df['dob']).dt.days // 365  #type: ignore

    # Gender
    gender_le = load_encoder('label_encoder_gender.pkl')
    df['gender_encoded'] = gender_le.transform(df['gender'])

    # Frequency Encodings
    for col in ['merchant', 'city', 'street', 'state', 'job']:
        mapping = load_encoder(f"{col}_mapping.pkl")
        df[f"{col}_encoded"] = df[col].map(mapping).fillna(0)

    # Category Mean Encoding
    category_mean = load_encoder("category_mean_mapping.pkl")
    df['category_encoded'] = df['category'].map(category_mean).fillna(0)

    # Time Features
    dt = pd.to_datetime(df['trans_date_trans_time'], unit='s')
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['month'] = dt.dt.month

    # Log amt
    df['log_amt'] = np.log(df['amt'] + 1)

    # Select only required features
    features = [
        'trans_date_trans_time', 'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'dob',
        'gender_encoded', 'merchant_encoded', 'city_encoded', 'street_encoded',
        'state_encoded', 'job_encoded', 'category_encoded',
        'hour', 'day_of_week', 'month', 'log_amt'
    ]

    return df[features]

def predict_transaction(input_dict):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    processed = preprocess_custom_input(input_dict)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Fraud" if prediction == 1 else "Not Fraud",
        "fraud_probability": round(probability, 4)
    }
