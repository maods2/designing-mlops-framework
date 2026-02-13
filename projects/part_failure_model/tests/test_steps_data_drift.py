"""Integration tests for PartFailureDataDrift step."""

import pytest

from steps.data_drift import PartFailureDataDrift


def test_data_drift_step_produces_drift_report(exec_context):
    """PartFailureDataDrift produces drift_report artifact with expected keys."""
    step = PartFailureDataDrift(exec_context)
    step.run()

    report = exec_context.storage.load("drift_report")
    assert report is not None
    assert "psi_score" in report
    assert "max_drift" in report
    assert "n_features" in report
    assert "drift_per_feature" in report
    assert isinstance(report["psi_score"], (int, float))
    assert isinstance(report["n_features"], int)
    assert report["n_features"] > 0


def test_data_drift_step_loads_reference_from_artifact(exec_context, synthetic_train_data):
    """Step uses train_data from storage as reference."""
    step = PartFailureDataDrift(exec_context)
    step.run()

    report = exec_context.storage.load("drift_report")
    # Reference has 10 features (from synthetic data)
    assert report["n_features"] == 10
    assert len(report["drift_per_feature"]) == 10
