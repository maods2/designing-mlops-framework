#!/usr/bin/env python
"""Test Dataproc/Spark execution format locally (no Spark cluster required)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

monorepo_root = Path(__file__).resolve().parent.parent
model_root = monorepo_root / "example_model"
sys.path.insert(0, str(monorepo_root))


def test_build_package():
    """Test building root.zip under example_model/dist."""
    from mlplatform.spark.packager import build_root_zip

    out = build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=monorepo_root / "test_dist",
    )
    assert out.exists()
    assert out.suffix == ".zip"
    print(f"✓ build_root_zip -> {out}")


def test_run_spark_main_direct():
    """Test spark/main.py with packages (simulates main.py --packages root.zip)."""
    from mlplatform.config.loader import load_workflow_config
    from mlplatform.spark.config_serializer import write_workflow_config
    from mlplatform.spark.packager import build_root_zip

    # Build package
    build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=monorepo_root / "test_dist",
    )
    root_zip = monorepo_root / "test_dist" / "root.zip"

    # Load config and write run_config for Spark main.py
    dag_path = monorepo_root / "mlplatform" / "tests" / "fixtures" / "legacy_training_dag.yaml"
    workflow = load_workflow_config(dag_path, config_dir=model_root / "config")
    model_cfg = workflow.models[0]

    config_path = monorepo_root / "test_dist" / "run_config.json"
    artifacts_dir = monorepo_root / "test_artifacts"
    write_workflow_config(
        workflow,
        model_cfg,
        config_path,
        base_path=str(artifacts_dir),
        version="test_v1",
    )

    # Create train input CSV
    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    train_csv = monorepo_root / "test_dist" / "train_input.csv"
    pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]).assign(target=y).to_csv(
        train_csv, index=False
    )

    # Run train step via spark/main.py (with packages)
    spark_main = monorepo_root / "mlplatform" / "mlplatform" / "spark" / "main.py"
    result = subprocess.run(
        [
            sys.executable,
            str(spark_main),
            "--config",
            str(config_path),
            "--packages",
            str(root_zip),
            "--project-root",
            str(monorepo_root),
        ],
        cwd=str(monorepo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"spark/main.py failed: {result.stderr}"
    print(f"✓ run_spark_step (train with root.zip)")


def test_local_spark_runner():
    """Test local-spark profile (SparkBatchInference) in direct mode."""
    from mlplatform.config.loader import load_workflow_config
    from mlplatform.runner import _build_context
    from mlplatform.runner.workflow import _run_prediction, _run_training

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    train_data = {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }

    artifacts_dir = monorepo_root / "test_artifacts"
    train_workflow = load_workflow_config(model_root / "pipeline" / "train.yaml")
    pred_workflow = load_workflow_config(model_root / "pipeline" / "predict.yaml")

    train_ctx = _build_context(
        train_workflow, train_workflow.models[0], "local-spark", "test_v1", str(artifacts_dir)
    )
    train_ctx.optional_configs["train_data"] = train_data

    _run_training(train_workflow.models[0], train_ctx)

    pred_ctx = _build_context(
        pred_workflow, pred_workflow.models[0], "local-spark", "test_v1", str(artifacts_dir)
    )
    from mlplatform.profiles.registry import get_profile

    inference = get_profile("local-spark").inference_strategy_factory()
    pred_result = _run_prediction(pred_workflow.models[0], pred_ctx, inference)
    assert pred_result is not None
    print("✓ LocalSparkRunner (direct mode)")


def main():
    print("Testing Spark/Dataproc local execution...\n")
    test_build_package()
    test_run_spark_main_direct()
    test_local_spark_runner()
    print("\nAll Spark local tests passed.")


if __name__ == "__main__":
    main()
