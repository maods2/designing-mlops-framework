#!/usr/bin/env python
"""Test the MLOps framework with template_model."""

import sys
from pathlib import Path

# Monorepo root; template_model is the model project
monorepo_root = Path(__file__).resolve().parent.parent
model_root = monorepo_root / "template_model"
sys.path.insert(0, str(monorepo_root))


def test_load_config():
    """Test loading pipeline config."""
    from mlplatform.local import load_pipeline_config

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    assert config.pipeline_name == "simple_pipeline"
    assert config.model_name == "simple_model"
    assert len(config.steps) == 2
    print("✓ load_pipeline_config")


def test_run_step_local():
    """Test running a single step (train)."""
    from mlplatform.local import run_step_local

    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    train_data = {"X": pd.DataFrame(X), "y": pd.Series(y)}

    result = run_step_local(
        step_name="train",
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
        project_root=model_root,
        train_data=train_data,
    )
    assert result is not None
    print("✓ run_step_local (train)")


def test_run_pipeline_local():
    """Test running full pipeline (train + inference)."""
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    train_data = {"X": pd.DataFrame(X), "y": pd.Series(y)}
    inference_df = pd.DataFrame(X[:10])

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    results = run_pipeline_local(
        config,
        project_root=model_root,
        base_path=model_root / "artifacts",
        train={"train_data": train_data},
        inference={"inference_data": inference_df},
    )
    assert "train" in results
    assert "inference" in results
    assert len(results["inference"]) == 10
    print("✓ run_pipeline_local")


def main():
    print("Testing MLOps framework with template_model...\n")
    test_load_config()
    test_run_step_local()
    test_run_pipeline_local()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
