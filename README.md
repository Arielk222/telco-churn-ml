# Telco Churn Prediction — End-to-End ML Project (FastAPI)

End-to-end churn prediction project using the Kaggle Telco Customer Churn dataset.
Includes feature engineering, XGBoost modeling, business threshold selection, SHAP explainability, and deployment via FastAPI.

## What this repo demonstrates
- Data cleaning & preprocessing
- Feature engineering
- Model training (XGBoost)
- Business-driven thresholding (precision/recall tradeoffs + ROI framing)
- Explainability (SHAP)
- Production-style inference service (FastAPI)
- Reproducible artifacts (model + feature columns + threshold)

## Quickstart
### 1) Create venv and install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"