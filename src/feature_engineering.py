import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def preprocess_features(df, save_encoders=True, encoder_path="models/encoders/"):
    """
    Perform feature engineering: encoding, time features, log transform.

    Args:
        df (pd.DataFrame): Raw dataframe
        save_encoders (bool): Whether to save encoders
        encoder_path (str): Path to save encoder mappings

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df = df.copy()

    # ---- Convert timestamp ----
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df['trans_date_trans_time'] = df['trans_date_trans_time'].astype(int) / 10**9

    # ---- Age from DOB ----
    df['dob'] = pd.to_datetime(df['dob'])
    df['dob'] = (pd.Timestamp.now() - df['dob']).dt.days // 365

    # ---- Label Encode 'gender' ----
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    if save_encoders:
        with open(f"{encoder_path}label_encoder_gender.pkl", 'wb') as f:
            pickle.dump(le_gender, f)

    # ---- Frequency Encoding ----
    for col in ['merchant', 'city', 'street', 'state', 'job']:
        counts = df[col].value_counts().to_dict()
        df[f"{col}_encoded"] = df[col].map(counts)
        if save_encoders:
            with open(f"{encoder_path}{col}_mapping.pkl", 'wb') as f:
                pickle.dump(counts, f)

    # ---- Mean Encoding 'category' ----
    category_mean = df.groupby('category')['is_fraud'].mean().to_dict()
    df['category_encoded'] = df['category'].map(category_mean)
    if save_encoders:
        with open(f"{encoder_path}category_mean_mapping.pkl", 'wb') as f:
            pickle.dump(category_mean, f)

    # ---- Log transform amount ----
    df['log_amt'] = np.log(df['amt'] + 1)

    return df
