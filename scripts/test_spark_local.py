#!/usr/bin/env python
"""Test Dataproc/Spark execution format locally (no Spark cluster required).

Uses example_model and the current mlplatform API. See docs/PYSPARK_RUNNING_ORDERS.md
for the canonical PySpark run reference.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

monorepo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(monorepo_root))
sys.path.insert(0, str(monorepo_root / "mlplatform"))


def test_build_package():
    """Test building root.zip for example_model."""
    from mlplatform.spark.packager import build_root_zip

    out = build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=monorepo_root / "test_spark_dist",
    )
    assert out.exists()
    assert out.suffix == ".zip"
    print(f"✓ build_root_zip -> {out}")


def test_run_spark_main_with_packages():
    """Test spark/main.py with --packages (simulates cloud-like execution)."""
    from mlplatform.config.factory import ConfigLoaderFactory
    from mlplatform.runner import _build_context
    from mlplatform.spark.config_serializer import write_workflow_config
    from mlplatform.spark.main import main as spark_main
    from mlplatform.spark.packager import build_root_zip

    artifacts_dir = monorepo_root / "test_spark_artifacts"
    dist_dir = monorepo_root / "test_spark_dist"
    dist_dir.mkdir(exist_ok=True)

    # 1. Train a model (required for prediction)
    train_pipeline = ConfigLoaderFactory.load_pipeline_config(
        monorepo_root / "example_model" / "pipeline" / "train.yaml",
        task_id="train_model",
        config_names=["global", "train-local"],
    )
    task_cfg = train_pipeline.tasks[0]
    ctx = _build_context(train_pipeline, task_cfg, "local", "test_spark_v1", str(artifacts_dir))

    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    ctx.optional_configs["train_data"] = {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }

    from example_model.train import MyTrainer
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()

    # 2. Build root.zip
    root_zip = build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=dist_dir,
    )

    # 3. Serialize prediction config
    pred_pipeline = ConfigLoaderFactory.load_pipeline_config(
        monorepo_root / "example_model" / "pipeline" / "predict.yaml",
        task_id="predict",
        config_names=["global", "predict-local"],
    )
    pred_task_cfg = pred_pipeline.tasks[0]
    config_path = dist_dir / "spark_config.json"
    write_workflow_config(
        pred_pipeline,
        pred_task_cfg,
        config_path,
        base_path=str(artifacts_dir),
        version="test_spark_v1",
    )

    # 4. Create input CSV
    X_test, _ = make_classification(n_samples=10, n_features=5, random_state=99)
    input_csv = dist_dir / "spark_input.csv"
    pd.DataFrame(X_test, columns=["f0", "f1", "f2", "f3", "f4"]).to_csv(input_csv, index=False)

    # 5. Run main.py with --packages (simulates cloud)
    output_path = dist_dir / "spark_predictions.parquet"
    old_argv = sys.argv
    try:
        sys.argv = [
            "spark_main",
            "--config", str(config_path),
            "--input-path", str(input_csv),
            "--output-path", str(output_path),
            "--packages", str(root_zip),
        ]
        exit_code = spark_main()
    finally:
        sys.argv = old_argv

    assert exit_code == 0, f"spark_main returned {exit_code}"
    result = pd.read_parquet(output_path)
    assert "prediction" in result.columns
    assert len(result) == 10
    print("✓ spark/main.py with --packages (cloud-like)")


def main():
    print("Testing Spark/Dataproc local execution...\n")
    test_build_package()
    test_run_spark_main_with_packages()
    print("\nAll Spark local tests passed.")


if __name__ == "__main__":
    main()
