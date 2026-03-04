from __future__ import annotations
from src.features import engineer_features
from src.preprocess import one_hot_transform

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
    df = engineer_features(df)
    X = one_hot_transform(df, bundle.feature_columns, drop_first=False)

    dmat = xgb.DMatrix(X)
    proba = float(bundle.booster.predict(dmat)[0])

    will_churn = proba >= bundle.threshold
    return proba, will_churn