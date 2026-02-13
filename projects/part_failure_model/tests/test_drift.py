"""Unit tests for custom.drift.compute_drift."""

import numpy as np
import pandas as pd
import pytest

from custom.drift import compute_drift


@pytest.fixture
def reference_df():
    """Reference (training) data."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 5),
        columns=["f0", "f1", "f2", "f3", "f4"],
    )


@pytest.fixture
def current_same(reference_df):
    """Current data identical to reference (no drift)."""
    return reference_df.copy()


@pytest.fixture
def current_drifted(reference_df):
    """Current data with shifted distribution (drift)."""
    cur = reference_df.copy()
    cur["f0"] = cur["f0"] + 5.0  # Strong mean shift
    cur["f1"] = cur["f1"] * 3.0  # Variance change
    return cur


def test_compute_drift_same_data_returns_low_scores(reference_df, current_same):
    """When reference and current are identical, drift scores should be near zero."""
    report = compute_drift(reference_df, current_same)
    assert "psi_score" in report
    assert "max_drift" in report
    assert "n_features" in report
    assert "drift_per_feature" in report
    assert report["n_features"] == 5
    assert report["psi_score"] < 0.1
    assert report["max_drift"] < 0.1


def test_compute_drift_drifted_data_returns_higher_scores(reference_df, current_drifted):
    """When current has drifted, psi_score and max_drift should be higher."""
    report = compute_drift(reference_df, current_drifted)
    assert report["psi_score"] > 0.5
    assert report["max_drift"] > 0.5


def test_compute_drift_accepts_train_data_dict(reference_df, current_same):
    """compute_drift accepts dict with X_train key."""
    train_data = {"X_train": reference_df, "y_train": None}
    report = compute_drift(train_data, current_same)
    assert "psi_score" in report
    assert report["n_features"] == 5


def test_compute_drift_accepts_columns_filter(reference_df, current_same):
    """columns parameter filters which features are used."""
    report = compute_drift(reference_df, current_same, columns=["f0", "f1"])
    assert report["n_features"] == 2
    assert len(report["drift_per_feature"]) == 2


def test_compute_drift_empty_numeric_returns_zero():
    """When no numeric columns, returns zero drift."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    report = compute_drift(df, df)
    assert report["psi_score"] == 0.0
    assert report["max_drift"] == 0.0
    assert report["n_features"] == 0


def test_compute_drift_output_types(reference_df, current_same):
    """Report values have expected types."""
    report = compute_drift(reference_df, current_same)
    assert isinstance(report["psi_score"], (int, float))
    assert isinstance(report["max_drift"], (int, float))
    assert isinstance(report["n_features"], int)
    assert isinstance(report["drift_per_feature"], dict)
