#!/usr/bin/env python
"""Test the MLOps framework with example_model."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

# Monorepo root; example_model is the model project
monorepo_root = Path(__file__).resolve().parent.parent
model_root = monorepo_root / "example_model"
sys.path.insert(0, str(monorepo_root))


def test_load_config():
    """Test loading workflow config."""
    from mlplatform.config.loader import load_workflow_config

    config = load_workflow_config(
        dag_path=model_root / "pipeline" / "train.yaml",
    )
    assert config.workflow_name == "example_workflow_sequential"
    assert config.pipeline_type == "training"
    assert len(config.models) >= 1
    assert config.models[0].model_name == "example_model"
    print("✓ load_workflow_config")


def test_run_step_local():
    """Test running a single step (train)."""
    from mlplatform.config.loader import load_workflow_config
    from mlplatform.runner import _build_context
    from mlplatform.runner.workflow import _run_training

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    train_data = {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }

    workflow = load_workflow_config(model_root / "pipeline" / "train.yaml")
    model_cfg = workflow.models[0]
    artifacts_dir = monorepo_root / "test_artifacts"
    ctx = _build_context(workflow, model_cfg, "local", "test_v1", str(artifacts_dir))
    ctx.optional_configs["train_data"] = train_data

    _run_training(model_cfg, ctx)
    model_path = artifacts_dir / workflow.feature_name / model_cfg.model_name / "test_v1" / "model.pkl"
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    print("✓ run_step_local (train)")


def test_run_pipeline_local():
    """Test running full pipeline (train + inference)."""
    from mlplatform.config.loader import load_workflow_config
    from mlplatform.runner import _build_context
    from mlplatform.runner import run_workflow
    from mlplatform.runner.workflow import _run_training

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    train_data = {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }

    artifacts_dir = monorepo_root / "test_artifacts"
    train_dag = model_root / "pipeline" / "train.yaml"
    pred_dag = model_root / "pipeline" / "predict.yaml"

    # Run training (with train_data override for deterministic test)
    workflow = load_workflow_config(train_dag)
    model_cfg = workflow.models[0]
    ctx = _build_context(workflow, model_cfg, "local", "test_v1", str(artifacts_dir))
    ctx.optional_configs["train_data"] = train_data
    _run_training(model_cfg, ctx)

    # Run prediction
    results_pred = run_workflow(
        pred_dag,
        profile="local",
        version="test_v1",
        base_path=str(artifacts_dir),
    )
    assert "example_model" in results_pred
    print("✓ run_pipeline_local")


def main():
    print("Testing MLOps framework with example_model...\n")
    test_load_config()
    test_run_step_local()
    test_run_pipeline_local()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
