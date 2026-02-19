"""Evaluation logic - load model and test data, compute metrics."""

import pandas as pd
from sklearn.metrics import accuracy_score

import joblib



def evaluate(
    model: str,
    df: pd.DataFrame,
    target_col: str = "target",
) -> dict:
    """Load model and test data, return accuracy."""
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    preds = model.predict(X)
    return {"accuracy": float(accuracy_score(y, preds))}


