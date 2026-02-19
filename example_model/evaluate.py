"""Evaluation logic - load model, scaler and test data, compute metrics."""

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score

import example_model.constants as cons


def evaluate(
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    target_col: str = "target",
) -> dict:
    """Load model and scaler, transform features, return accuracy."""
    feature_cols = [c for c in cons.FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols] if feature_cols else df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return {"accuracy": float(accuracy_score(y, preds))}
