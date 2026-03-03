from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df["total_value"] = df["tenure"] * df["MonthlyCharges"]
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace({0: np.nan})
    df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(0.0)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def is_yes(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.lower() == "yes").astype(int)

    df["num_services"] = sum(is_yes(df[c]) for c in service_cols)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 10_000],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
    ).astype(str)

    df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)

    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}\n"
            "Put the Kaggle CSV at data/Telco-Customer-Churn.csv (data/ is gitignored)."
        )

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = (df["Churn"].astype(str) == "Yes").astype(int)
    X_raw = df.drop(columns=["Churn", "customerID"], errors="ignore")

    X = pd.get_dummies(X_raw, drop_first=False)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC: {auc:.4f}")

    threshold = 0.415  # swap later with ROI-optimal threshold

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ✅ Stable format: XGBoost Booster JSON
    model.get_booster().save_model(str(ARTIFACTS / "xgb_booster.json"))

    # ✅ Keep these as joblib (fine)
    joblib.dump(feature_columns, ARTIFACTS / "feature_columns.pkl")
    joblib.dump(threshold, ARTIFACTS / "decision_threshold.pkl")

    metadata = {
        "python": "3.11",
        "xgboost": xgb.__version__,
        "roc_auc": float(auc),
        "threshold": float(threshold),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved artifacts: xgb_booster.json, feature_columns.pkl, decision_threshold.pkl, metadata.json")


if __name__ == "__main__":
    main()