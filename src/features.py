from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


SERVICE_COLS: Iterable[str] = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared feature engineering used by BOTH training and serving.

    Keep this as the single source of truth.
    """
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
    def is_yes(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.lower() == "yes").astype(int)

    df["num_services"] = 0
    for c in SERVICE_COLS:
        if c in df.columns:
            df["num_services"] += is_yes(df[c])

    # Tenure buckets (must match training + serving)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 10_000],
        labels=["0-12", "13-24", "25-48", "49-60", "60+"],
    ).astype(str)

    df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)

    return df