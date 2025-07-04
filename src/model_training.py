import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_model(df, model_path="models/fraud_detection_model.pkl"):
    """
    Trains an XGBoost model to detect fraud.

    Args:
        df (pd.DataFrame): Preprocessed dataframe
        model_path (str): Where to save the trained model

    Returns:
        model, metrics_dict
    """

    # ---- Feature Selection ----
    features = [
        'trans_date_trans_time', 'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'dob',
        'gender_encoded', 'merchant_encoded', 'city_encoded', 'street_encoded',
        'state_encoded', 'job_encoded', 'category_encoded',
        'hour', 'day_of_week', 'month', 'log_amt'
    ]
    target = 'is_fraud'

    X = df[features]
    y = df[target]

    # ---- SMOTE for Imbalance ----
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)  #type: ignore

    # ---- Train-Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # ---- Train XGBoost Model ----
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }

    # ---- Save Model ----
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model, metrics
