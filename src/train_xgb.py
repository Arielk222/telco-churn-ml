from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import engineer_features
from src.preprocess import one_hot_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "configs" / "train_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


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

    prep = one_hot_fit(X_raw, drop_first=False)
    X = prep.X
    feature_columns = prep.feature_columns

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())
    scale_pos_weight = neg / max(pos, 1)

    model_params = dict(config.get("model", {}))
    model_params.pop("scale_pos_weight", None)  # computed dynamically

    model = xgb.XGBClassifier(
        **model_params,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC: {auc:.4f}")

    threshold = float(config.get("threshold", 0.415))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model.get_booster().save_model(str(ARTIFACTS / "xgb_booster.json"))
    joblib.dump(feature_columns, ARTIFACTS / "feature_columns.pkl")
    joblib.dump(threshold, ARTIFACTS / "decision_threshold.pkl")

    metadata = {
        "python": "3.11",
        "xgboost": xgb.__version__,
        "roc_auc": float(auc),
        "threshold": float(threshold),
        "config_path": str(CONFIG_PATH),
        "model_params": model_params,
        "split": {"test_size": test_size, "random_state": random_state},
        "scale_pos_weight": float(scale_pos_weight),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        "Saved artifacts: xgb_booster.json, feature_columns.pkl, decision_threshold.pkl, metadata.json"
    )


if __name__ == "__main__":
    main()