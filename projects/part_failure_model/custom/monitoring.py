"""Model health monitoring for part-failure pipeline.

Data scientists extend this module with custom monitoring logic.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd


def compute_model_health(
    model: Any,
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Compute monitoring metrics: accuracy, prediction distribution stats.

    Args:
        model: Trained model with predict/predict_proba
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Dict with monitoring_report: accuracy, prediction_mean, etc.
    """
    from custom.evaluation import compute_metrics

    y_pred = model.predict(X_val)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val)
        if proba.shape[1] >= 2:
            y_proba = proba[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_proba)

    pred_mean = float(np.mean(y_pred))
    pred_std = float(np.std(y_pred)) if len(y_pred) > 1 else 0.0
    metrics["prediction_mean"] = pred_mean
    metrics["prediction_std"] = pred_std
    metrics["n_samples"] = len(y_val)

    return metrics
