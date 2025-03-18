## **AI Fraud Detection System Using XGBoost**

This project leverages **XGBoost** to build a high-accuracy fraud detection system that identifies fraudulent credit card transactions. The model has been trained on real-world credit card transaction data and achieves an impressive performance with:  
✅ **Train Accuracy:** 99.85%  
✅ **Test Accuracy:** 99.83%  
✅ **Precision:** 99.81%  
✅ **Recall:** 99.85%  
✅ **F1-Score:** 99.83%  

---

## 📚 **Dataset Overview**

**Dataset Link** => [Credit Card Fraud Dataset](https://gts.ai/dataset-download/credit-card-transactions-dataset/)
The dataset contains transaction-level details such as:  
- **Transaction Amount (`amt`)**  
- **Merchant Information (`merchant`, `category`)**  
- **Geolocation Data (`lat`, `long`, `merch_lat`, `merch_long`)**  
- **Timestamp (`unix_time`)**  
- **Fraud Indicator (`is_fraud`)**  

---

## 🛠️ **Project Structure**
```
├── fraud_detection_xgb.pkl             # Trained XGBoost model
├── fraud_detection.ipynb               # Main notebook for model training
├── requirements.txt                    # Required dependencies
└── README.md                           # Project documentation
```


### 1 **Clone the Repository**
```bash
git clone https://github.com/yourusername/ai-fraud-detection.git
cd ai-fraud-detection
```

---

### 2 **Create Virtual Environment & Install Dependencies**
```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
# venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 3 **Run the Jupyter Notebook (If You Want To Train Model)**
```bash
# Open the notebook to train and save the model
jupyter notebook
```
- Open `fraud_detection.ipynb` and run all the cells to:
=> Clean and preprocess the data  
=> Train the XGBoost model  
=> Save the model as `fraud_detection_xgb.pkl`  

---

## **Model Performance**
- **Accuracy:** 99.83%  
- **Precision:** 99.81%  
- **Recall:** 99.85%  
- **F1-Score:** 99.83%  
- **ROC-AUC:** 99.83%  

---
