# Telco Churn Prediction — End-to-End ML Project (FastAPI)

End-to-end churn prediction project using the Kaggle **Telco Customer Churn** dataset.  
Includes feature engineering, XGBoost modeling, business-driven threshold selection (precision/recall ↔ ROI), SHAP explainability, and deployment via FastAPI.

## What this repo demonstrates
- Data cleaning & preprocessing
- Feature engineering (shared across training + serving)
- Model training (XGBoost)
- Threshold tuning based on business tradeoffs + ROI framing
- Explainability (SHAP)
- Production-style inference service (FastAPI)
- Reproducible artifacts

## Repo structure
- `src/` — training + feature engineering + preprocessing
- `app/` — FastAPI inference service
- `notebooks/` — EDA / modeling / ROI / SHAP narrative
- `artifacts/` — model + feature columns + threshold + metadata
- `reports/` — saved figures/tables 

## Quickstart (run the API locally)
### 1) Create venv + install

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"


### 2) Start the API

uvicorn app.main:app --reload

# Health endpoint:
http://127.0.0.1:8000/health


# Swagger docs:
http://127.0.0.1:8000/docs

# Example Prediction

curl -X POST "http://127.0.0.1:8000/predict
"
-H "Content-Type: application/json"
-d '{
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 5,
"PhoneService": "Yes",
"MultipleLines": "No",
"InternetService": "Fiber optic",
"OnlineSecurity": "No",
"OnlineBackup": "Yes",
"DeviceProtection": "No",
"TechSupport": "No",
"StreamingTV": "Yes",
"StreamingMovies": "No",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 89.1,
"TotalCharges": 445.5
}'

# Example response:
{
"churn_probability": 0.876,
"threshold": 0.415,
"will_churn": true
}

# Training the Model

Download the Kaggle dataset and place the CSV at: [data/Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# Then run:

python -m src.train_xgb


# This will generate artifacts in: artifacts/


- `xgb_booster.json`
- `feature_columns.pkl`
- `decision_threshold.pkl`
- `metadata.json`

---

# Notebook

See the notebook for full analysis and model development:

notebooks/02_modeling_thresholds_shap.ipynb

The notebook includes:

- Exploratory data analysis
- Feature engineering exploration
- Model comparison
- Threshold optimization for business ROI
- SHAP explainability

---

# Notes on Model Portability

The model is saved as an **XGBoost Booster JSON file** (`xgb_booster.json`) instead of a pickled sklearn wrapper.

This avoids common version compatibility issues when deploying models.

---

# Future Improvements

- Model monitoring
- Retraining pipeline
- Docker containerization
- CI/CD testing pipeline
