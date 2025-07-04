# AI Fraud Detection System Using XGBoost

This project is a high-performance, modular AI system that detects **fraudulent credit card transactions** using **XGBoost** and the model has been trained on real-world credit card transaction data and achieves an impressive performance.

---

## Features

- **Model**: XGBoost Classifier
- **Techniques**: SMOTE for class imbalance, feature engineering, frequency & mean encoding
- **Architecture**: Modular Python scripts with `main.py` pipeline
- **Production-Ready**: Encoders, model pickling, custom JSON input testing
- **Performance**:
  - **Train Accuracy**: 99.85%
  - **Test Accuracy**: 99.83%
  - **Precision**: 99.81%
  - **Recall**: 99.85%
  - **F1-Score**: 99.83%
  - **ROC-AUC**: 99.83%

---

## Dataset Overview

**Dataset Download:** [Credit Card Transactions Dataset (GTS AI)](https://gts.ai/dataset-download/credit-card-transactions-dataset/)

The dataset contains transaction-level details such as:
- `amt` — Transaction amount
- `merchant`, `category` — Merchant & category info
- `lat`, `long`, `merch_lat`, `merch_long` — Geolocation
- `dob`, `gender`, `job`, `city_pop`, etc.
- `is_fraud` — Binary fraud label

---

## Project Structure

```

fraud-detection-project/
│
├── data/                        # Raw dataset (CSV)
├── models/                      # Trained model & encoders (ignored in .git)
│   └── fraud\_detection\_model.pkl
│
├── src/                         # Source modules
│   ├── data\_loader.py
│   ├── feature\_engineering.py
│   ├── model\_training.py
│   └── predict.py
│
├── main.py                      # Complete training + testing pipeline
├── requirements.txt
└── README.md

````

```

## Getting Started

###bClone the Repository
```bash
git clone https://github.com/AryakumarMishra/AI-Fraud-Detection.git
cd AI-Fraud-Detection
````

```

### Create Virtual Environment & Install Requirements

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```



### Run the Full Pipeline

```bash
python main.py
```

This will:

* Load the dataset from `data/`
* Perform feature engineering & save encoders
* Train the XGBoost model with SMOTE
* Save the model to `models/fraud_detection_model.pkl`
* Run a sample fraud prediction from a custom transaction

---

## Sample Output

```bash
Model Performance Metrics:
accuracy: 0.9985
precision: 0.9982
recall: 0.9989
f1_score: 0.9985
roc_auc: 0.9985

Running sample prediction...
Prediction: {'prediction': 0, 'label': 'Not Fraud', 'fraud_probability': 0.1346}
```
