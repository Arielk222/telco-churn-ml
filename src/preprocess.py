from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PreprocessResult:
    X: pd.DataFrame
    feature_columns: List[str]


def one_hot_fit(df: pd.DataFrame, drop_first: bool = False) -> PreprocessResult:
    """
    Fit-time one-hot encoding:
    - runs get_dummies
    - returns X and the learned feature columns (order matters)
    """
    X = pd.get_dummies(df, drop_first=drop_first)
    feature_columns = list(X.columns)
    return PreprocessResult(X=X, feature_columns=feature_columns)


def one_hot_transform(
    df: pd.DataFrame, feature_columns: List[str], drop_first: bool = False
) -> pd.DataFrame:
    """
    Serve-time transform:
    - runs get_dummies
    - aligns exactly to feature_columns (adds missing, drops extras, orders columns)
    """
    X = pd.get_dummies(df, drop_first=drop_first)

    # Add missing cols
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0

    # Drop unexpected cols
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)

    # Reorder
    return X[feature_columns]