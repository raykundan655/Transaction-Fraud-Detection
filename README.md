# Financial Fraud Detection

An intermediate-level Machine Learning project to detect fraudulent mobile money transactions using the PaySim dataset.

---

## Project Structure

```text
Financial-Fraud-Detection/
│
├── data/
│   └── paysim.csv
│
├── fraud_detection_model.py
├── app.py
├── fraud_model.pkl
└── requirements.txt
```

---

## Overview

- **Dataset:** PaySim Synthetic Mobile Money Transactions (~6.36 million rows).
- **Target Variable:** `isFraud` (1 = Fraudulent, 0 = Legitimate).
- **Class Imbalance:** Fraud accounts for ~0.129% of transactions (~1 fraud per 773 legitimate).
- **Feature Engineering:**
  - `orig_balance_diff`: Discrepancy between sender expected balance change and actual balance change.
  - `dest_balance_diff`: Discrepancy between receiver expected balance change and actual balance change.
  - `is_orig_zero`: Binary flag indicating if the sender's account was drained to zero.
  - `amount_to_oldbalance_orig`: Ratio of transaction amount relative to the sender's starting balance.
- **Preprocessing:** Scikit-learn `ColumnTransformer` with `SimpleImputer` and `StandardScaler` for numerical columns, and `OneHotEncoder` for transaction type.
- **Model:** XGBoost Classifier with `scale_pos_weight` to address class imbalance.
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 3-fold `StratifiedKFold` on a stratified sample, evaluated with Average Precision (PR-AUC).
- **Web App:** Clean, minimal Streamlit interface for instant transaction risk prediction.

---

## Model Evaluation Results

Evaluated on 1,272,524 untouched test records (20% test split):

- **Accuracy:** 99.999%
- **Precision:** 99.51%
- **Recall:** 99.76%
- **F1-Score:** 99.64%
- **ROC-AUC:** 0.9998
- **PR-AUC:** 0.9987

---

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python fraud_detection_model.py
```
This script loads the dataset, performs exploratory data analysis, cleans data, creates engineered features, tunes hyperparameters on a stratified sample, fits the final pipeline on the complete training set, prints evaluation metrics, and saves `fraud_model.pkl`.

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, enter transaction values, and click **Predict Fraud** for an immediate prediction.
