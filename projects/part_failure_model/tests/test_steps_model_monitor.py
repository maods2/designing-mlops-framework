"""Integration tests for PartFailureModelMonitor step."""

import pytest

from steps.model_monitor import PartFailureModelMonitor


def test_model_monitor_step_produces_monitoring_report(exec_context):
    """PartFailureModelMonitor produces monitoring_report artifact with expected keys."""
    step = PartFailureModelMonitor(exec_context)
    step.run()

    report = exec_context.storage.load("monitoring_report")
    assert report is not None
    assert "accuracy" in report
    assert "n_samples" in report
    assert "prediction_mean" in report
    assert "prediction_std" in report
    assert isinstance(report["accuracy"], (int, float))
    assert 0 <= report["accuracy"] <= 1
    assert report["n_samples"] > 0


def test_model_monitor_step_loads_model_from_artifact(exec_context, synthetic_train_data):
    """Step uses model from storage and produces valid metrics."""
    step = PartFailureModelMonitor(exec_context)
    step.run()

    report = exec_context.storage.load("monitoring_report")
    assert "f1_score" in report
    assert "precision" in report
    assert "recall" in report
