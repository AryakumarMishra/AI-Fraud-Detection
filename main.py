"""
MAKE SURE TO HAVE THE DATASET DOWNLOADED AND SAVED IN THE 'data/' FOLDER, AS MENTIONED IN 'data_path' BELOW.

ALSO MAKE SURE TO CREATE THE 'models' FOLDER AND ANOTHER 'encoders' FOLDER INSIDE 'models/' FOLDER. (EMPTY FOLDERS, ONCE YOU RUN THE FOLLOWING PIPELINE (>python main.py), THE MODELS AND NECESSARY ENCODERS WILL BE SAVED ACCORDINGLY)

"""


from src.data_loader import load_data
from src.feature_engineering import preprocess_features
from src.model_training import train_model
from src.predict import predict_transaction
import os

def main():
    print("\nAI Fraud Detection Pipeline Started")

    # === 1. Load Data ===
    data_path = "data/credit_card_transactions.csv"
    if not os.path.exists(data_path):
        print("Dataset not found at:", data_path)
        return
    print("Data loaded")
    df = load_data(data_path)

    # === 2. Feature Engineering ===
    print("Running feature engineering & saving encoders...")
    df = preprocess_features(df, save_encoders=True, encoder_path="models/encoders/")

    # === 3. Train Model ===
    print("Training XGBoost Model...")
    model, metrics = train_model(df, model_path="models/fraud_detection_model.pkl")

    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # === 4. Sample Prediction ===
    print("\nRunning sample prediction...")
    sample_input = {
        "trans_date_trans_time": "2023-06-01 14:30:00",
        "cc_num": 1234567890123456,
        "amt": 150.75,
        "zip": 10001,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000,
        "dob": "1995-05-15",
        "gender": "F",
        "merchant": "XYZ Store",
        "city": "New York",
        "street": "Main St",
        "state": "NY",
        "job": "Engineer",
        "category": "shopping_net"
    }

    prediction_output = predict_transaction(sample_input)
    print("Prediction:", prediction_output)

    print("\nPipeline execution completed.\n")

if __name__ == "__main__":
    main()
