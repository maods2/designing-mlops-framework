#!/usr/bin/env python
"""Test example_model against the simplified mlplatform API."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

monorepo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(monorepo_root))
sys.path.insert(0, str(monorepo_root / "mlplatform"))


def test_run_training_workflow():
    """Test running training via run_workflow with a pipeline YAML."""
    from mlplatform.runner import run_workflow, _build_context
    from mlplatform.config.factory import ConfigLoaderFactory

    dag_path = monorepo_root / "example_model" / "pipeline" / "train.yaml"
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        dag_path, task_id="train_model", config_names=["global", "train-local"]
    )

    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    train_data = {
        "X": pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]),
        "y": pd.Series(y),
    }

    artifacts_dir = monorepo_root / "test_artifacts"

    task_cfg = pipeline.tasks[0]
    ctx = _build_context(pipeline, task_cfg, "local", "test_v1", str(artifacts_dir))
    ctx.optional_configs["train_data"] = train_data

    from example_model.train import MyTrainer
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()

    # Standardized path: {base}/artifacts/{feature}/{model_train_version}/model/{artifact}
    model_train_version = f"{pipeline.feature_name}_{task_cfg.model_name}_train_test_v1"
    model_path = artifacts_dir / "artifacts" / pipeline.feature_name / model_train_version / "model" / "model.pkl"
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    print("PASS test_run_training_workflow")


def test_run_prediction():
    """Test running prediction after training."""
    from mlplatform.config.factory import ConfigLoaderFactory
    from mlplatform.runner import _build_context

    dag_path = monorepo_root / "example_model" / "pipeline" / "predict.yaml"
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        dag_path, task_id="predict", config_names=["global", "predict-local"]
    )
    task_cfg = pipeline.tasks[0]

    artifacts_dir = monorepo_root / "test_artifacts"
    ctx = _build_context(pipeline, task_cfg, "local", "test_v1", str(artifacts_dir))

    from example_model.predict import MyPredictor
    predictor = MyPredictor()
    predictor.context = ctx
    predictor.load_model()

    X_test, _ = make_classification(n_samples=5, n_features=5, random_state=99)
    test_df = pd.DataFrame(X_test, columns=["f0", "f1", "f2", "f3", "f4"])
    result = predictor.predict(test_df)

    assert "prediction" in result.columns, "Missing prediction column"
    assert len(result) == 5
    print("PASS test_run_prediction")


def test_build_package():
    """Test building root.zip for example_model."""
    from mlplatform.spark.packager import build_root_zip

    out = build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=monorepo_root / "test_dist",
    )
    assert out.exists()
    assert out.suffix == ".zip"
    print(f"PASS test_build_package -> {out}")


def test_config_serializer():
    """Test config serialization for Spark main.py consumption."""
    from mlplatform.config.factory import ConfigLoaderFactory
    from mlplatform.spark.config_serializer import write_workflow_config

    dag_path = monorepo_root / "example_model" / "pipeline" / "train.yaml"
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        dag_path, task_id="train_model", config_names=["global", "train-local"]
    )
    task_cfg = pipeline.tasks[0]

    config_path = monorepo_root / "test_dist" / "run_config.json"
    write_workflow_config(pipeline, task_cfg, config_path, base_path=str(monorepo_root / "test_artifacts"), version="test_v1")
    assert config_path.exists()

    import json
    with open(config_path) as f:
        data = json.load(f)
    assert data["runtime_config"]["feature_name"] == "example"
    assert data["runtime_config"]["model_name"] == "example_model"
    print("PASS test_config_serializer")


def test_pyspark_batch_prediction():
    """Test PySpark local batch prediction via spark/main.py mapInPandas."""
    import json
    from mlplatform.config.factory import ConfigLoaderFactory
    from mlplatform.spark.config_serializer import write_workflow_config
    from mlplatform.spark.main import _run_spark_inference

    artifacts_dir = monorepo_root / "test_artifacts"

    pipeline = ConfigLoaderFactory.load_pipeline_config(
        monorepo_root / "example_model" / "pipeline" / "predict.yaml",
        task_id="predict",
        config_names=["global", "predict-local"],
    )
    task_cfg = pipeline.tasks[0]
    config_path = monorepo_root / "test_dist" / "spark_pred_config.json"
    write_workflow_config(pipeline, task_cfg, config_path, base_path=str(artifacts_dir), version="test_v1")

    X_test, _ = make_classification(n_samples=10, n_features=5, random_state=99)
    csv_path = monorepo_root / "test_dist" / "spark_input.csv"
    pd.DataFrame(X_test, columns=["f0", "f1", "f2", "f3", "f4"]).to_csv(csv_path, index=False)

    output_path = str(monorepo_root / "test_dist" / "spark_predictions.parquet")
    with open(config_path) as f:
        config = json.load(f)
    _run_spark_inference(
        config,
        input_path=str(csv_path),
        output_path=output_path,
    )

    result = pd.read_parquet(output_path)
    assert "prediction" in result.columns, "Missing prediction column"
    assert len(result) == 10, f"Expected 10 rows, got {len(result)}"
    print("PASS test_pyspark_batch_prediction")


def main():
    print("Testing example_model with simplified mlplatform API...\n")
    test_run_training_workflow()
    test_run_prediction()
    test_build_package()
    test_config_serializer()
    test_pyspark_batch_prediction()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
