"""Unit tests for custom.monitoring.compute_model_health."""

import numpy as np
import pandas as pd
import pytest

from custom.monitoring import compute_model_health


@pytest.fixture
def mock_model():
    """Mock model with predict and predict_proba."""
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))

        def predict_proba(self, X):
            n = len(X)
            p = np.random.rand(n, 2)
            p /= p.sum(axis=1, keepdims=True)
            return p

    return MockModel()


@pytest.fixture
def X_val():
    """Validation features."""
    np.random.seed(99)
    return pd.DataFrame(
        np.random.randn(50, 5),
        columns=["f0", "f1", "f2", "f3", "f4"],
    )


@pytest.fixture
def y_val():
    """Validation labels."""
    np.random.seed(99)
    return pd.Series(np.random.randint(0, 2, 50))


def test_compute_model_health_returns_dict(mock_model, X_val, y_val):
    """compute_model_health returns a dict with expected keys."""
    report = compute_model_health(mock_model, X_val, y_val)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "precision" in report
    assert "recall" in report
    assert "f1_score" in report
    assert "prediction_mean" in report
    assert "prediction_std" in report
    assert "n_samples" in report


def test_compute_model_health_n_samples_matches_input(mock_model, X_val, y_val):
    """n_samples equals len(y_val)."""
    report = compute_model_health(mock_model, X_val, y_val)
    assert report["n_samples"] == len(y_val)


def test_compute_model_health_accuracy_in_range(mock_model, X_val, y_val):
    """Accuracy is between 0 and 1."""
    report = compute_model_health(mock_model, X_val, y_val)
    assert 0 <= report["accuracy"] <= 1


def test_compute_model_health_with_real_model(trained_model, synthetic_train_data):
    """Works with actual PartFailureModel and produces sensible metrics."""
    X_val = synthetic_train_data["X_val"]
    y_val = synthetic_train_data["y_val"]
    report = compute_model_health(trained_model, X_val, y_val)
    assert "accuracy" in report
    assert "roc_auc" in report
    assert report["n_samples"] == len(y_val)
    assert 0 <= report["accuracy"] <= 1
    assert 0 <= report["prediction_mean"] <= 1
    assert report["prediction_std"] >= 0


def test_compute_model_health_accepts_numpy_arrays(mock_model, X_val, y_val):
    """Accepts numpy arrays instead of DataFrames."""
    report = compute_model_health(mock_model, X_val.values, y_val.values)
    assert "accuracy" in report
    assert report["n_samples"] == len(y_val)
