"""Evaluation logic - reusable from train.py or standalone."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import model_code.constants as cons


def evaluate(
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, float]:
    """Score a model on a labelled DataFrame and return metrics.

    Args:
        model: Fitted sklearn estimator with ``.predict()``.
        scaler: Fitted scaler with ``.transform()``.
        df: DataFrame containing feature columns and *target_col*.
        target_col: Name of the ground-truth column.

    Returns:
        Dict of metric name -> value.
    """
    feature_cols = [c for c in cons.FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols] if feature_cols else df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
