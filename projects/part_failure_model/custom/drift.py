"""Data drift detection for part-failure pipeline.

Data scientists extend this module with custom drift logic (e.g. Evidently, Alibi Detect).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def compute_drift(
    reference_df: Union[pd.DataFrame, dict],
    current_df: Union[pd.DataFrame, dict],
    columns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute drift metrics (e.g. PSI, mean shift) between reference and current data.

    Args:
        reference_df: Reference/training data (DataFrame or train_data dict with X_train)
        current_df: Current/production data (DataFrame or dict)
        columns: Feature columns to compute drift for (default: all numeric)

    Returns:
        Dict with drift_report: psi_score, max_drift, per_feature stats
    """
    if isinstance(reference_df, dict):
        ref = reference_df.get("X_train")
        if ref is None:
            ref = pd.DataFrame(reference_df)
    else:
        ref = reference_df
    if isinstance(current_df, dict):
        cur = current_df.get("X_train") or current_df.get("current")
        if cur is None:
            cur = pd.DataFrame(current_df)
    else:
        cur = current_df

    if not isinstance(ref, pd.DataFrame):
        ref = pd.DataFrame(ref)
    if not isinstance(cur, pd.DataFrame):
        cur = pd.DataFrame(cur)

    numeric_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]
    if not numeric_cols:
        return {"psi_score": 0.0, "max_drift": 0.0, "n_features": 0}

    drift_scores = []
    for col in numeric_cols:
        if col not in cur.columns:
            continue
        ref_vals = ref[col].dropna()
        cur_vals = cur[col].dropna()
        if len(ref_vals) < 2 or len(cur_vals) < 2:
            drift_scores.append(0.0)
            continue
        mean_diff = abs(ref_vals.mean() - cur_vals.mean())
        std_ref = ref_vals.std()
        std_cur = cur_vals.std()
        if std_ref > 0 and std_cur > 0:
            psi_like = mean_diff / (0.5 * (std_ref + std_cur) + 1e-8)
        else:
            psi_like = mean_diff
        drift_scores.append(float(min(psi_like, 10.0)))

    max_drift = max(drift_scores) if drift_scores else 0.0
    psi_score = sum(drift_scores) / len(drift_scores) if drift_scores else 0.0

    return {
        "psi_score": round(psi_score, 4),
        "max_drift": round(max_drift, 4),
        "n_features": len(drift_scores),
        "drift_per_feature": dict(zip(numeric_cols[: len(drift_scores)], [round(s, 4) for s in drift_scores])),
    }
