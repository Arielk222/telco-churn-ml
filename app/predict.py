from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(frozen=True)
class ModelBundle:
    booster: xgb.Booster
    feature_columns: List[str]
    threshold: float


def load_bundle(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelBundle:
    # Load stable booster model (version-robust compared to pickled sklearn wrapper)
    booster = xgb.Booster()
    booster.load_model(str(artifacts_dir / "xgb_booster.json"))

    feature_columns = joblib.load(artifacts_dir / "feature_columns.pkl")
    threshold = joblib.load(artifacts_dir / "decision_threshold.pkl")

    # Normalize threshold if saved as dict
    if isinstance(threshold, dict):
        threshold = float(threshold.get("threshold"))
    return ModelBundle(
        booster=booster,
        feature_columns=list(feature_columns),
        threshold=float(threshold),
    )


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Engineered features
    df["total_value"] = df["tenure"] * df["MonthlyCharges"]
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace({0: np.nan})
    df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(0.0)

    # Service count
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

    df["num_services"] = sum(is_yes(df[c]) for c in service_cols if c in df.columns)

    # Tenure buckets (must match training logic!)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 10_000],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
    ).astype(str)

    df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)

    return df


def _one_hot_align(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X = pd.get_dummies(df, drop_first=False)

    # Add missing columns
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0

    # Drop extra columns not seen in training
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)

    # Ensure exact column order
    X = X[feature_columns]
    return X


def predict_one(bundle: ModelBundle, payload: Dict[str, Any]) -> Tuple[float, bool]:
    df = pd.DataFrame([payload])
    df = _engineer_features(df)
    X = _one_hot_align(df, bundle.feature_columns)

    dmat = xgb.DMatrix(X)
    proba = float(bundle.booster.predict(dmat)[0])

    will_churn = proba >= bundle.threshold
    return proba, will_churn