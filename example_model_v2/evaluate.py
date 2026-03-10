"""Evaluation logic for trend regression - reusable from train.py or standalone."""

from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import example_model_v2.constants as cons


def evaluate(
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, float]:
    """Score a trend regression model on a labelled DataFrame and return metrics.

    Args:
        model: Fitted sklearn estimator with ``.predict()``.
        scaler: Fitted scaler with ``.transform()``.
        df: DataFrame containing feature columns and *target_col*.
        target_col: Name of the ground-truth column.

    Returns:
        Dict of metric name -> value (MSE, MAE, R2).
    """
    feature_cols = [c for c in cons.FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols] if feature_cols else df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return {
        "mse": float(mean_squared_error(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }
