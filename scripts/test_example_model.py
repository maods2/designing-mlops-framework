#!/usr/bin/env python
"""Test example_model: procedural and Spark batch prediction."""

import sys
from pathlib import Path

monorepo_root = Path(__file__).resolve().parent.parent
model_root = monorepo_root / "example_model"
sys.path.insert(0, str(monorepo_root))


def test_load_data_csv():
    """Test framework load_data with CSV."""
    from mlplatform.data import load_data

    csv_path = model_root / "data" / "sample_inference.csv"
    df = load_data({"type": "csv", "path": str(csv_path)}, format="pandas")
    assert len(df) == 3
    assert "f0" in df.columns
    print("✓ load_data (CSV)")


def test_load_config():
    """Test loading pipeline config."""
    from mlplatform.local import load_pipeline_config

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    assert config.pipeline_name == "example_pipeline"
    assert config.model_name == "example_model"
    assert len(config.steps) == 2  # train, inference
    print("✓ load_pipeline_config")


def test_run_pipeline_local():
    """Test running full pipeline (train + inference) with CSV."""
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    import pandas as pd

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    train_csv = model_root / "data" / "sample_train.csv"
    inference_csv = model_root / "data" / "sample_inference.csv"
    train_df = pd.read_csv(train_csv)
    train_data = {"X": train_df.drop(columns=["target"]), "y": train_df["target"]}
    results = run_pipeline_local(
        config,
        project_root=model_root,
        base_path=model_root / "artifacts",
        train={"train_data": train_data},
        inference={"inference_data": pd.read_csv(inference_csv)},
    )
    assert "train" in results
    assert "inference" in results
    assert len(results["inference"]) == 3
    print("✓ run_pipeline_local (train + inference)")


def test_procedural_prediction_csv():
    """Test procedural prediction with CSV."""
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    import pandas as pd

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    train_csv = model_root / "data" / "sample_train.csv"
    inference_csv = model_root / "data" / "sample_inference.csv"
    train_df = pd.read_csv(train_csv)
    train_data = {"X": train_df.drop(columns=["target"]), "y": train_df["target"]}
    results = run_pipeline_local(
        config,
        project_root=model_root,
        base_path=model_root / "artifacts",
        train={"train_data": train_data},
        inference={"inference_data": pd.read_csv(inference_csv)},
    )
    assert len(results["inference"]) == 3
    print("✓ procedural prediction (CSV)")


def test_procedural_prediction_load_data():
    """Test procedural prediction using load_data (framework data retrieval)."""
    from mlplatform.data import load_data
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    train_csv = model_root / "data" / "sample_train.csv"
    inference_csv = model_root / "data" / "sample_inference.csv"
    train_df = load_data({"type": "csv", "path": str(train_csv)}, format="pandas")
    train_data = {"X": train_df.drop(columns=["target"]), "y": train_df["target"]}
    inference_df = load_data({"type": "csv", "path": str(inference_csv)}, format="pandas")
    results = run_pipeline_local(
        config,
        project_root=model_root,
        base_path=model_root / "artifacts",
        train={"train_data": train_data},
        inference={"inference_data": inference_df},
    )
    assert len(results["inference"]) == 3
    print("✓ procedural prediction (load_data)")


def test_local_spark_runner():
    """Test LocalSparkRunner in direct mode (in-process, no packaging)."""
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    import pandas as pd

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="local_spark",
    )
    train_csv = model_root / "data" / "sample_train.csv"
    inference_csv = model_root / "data" / "sample_inference.csv"
    train_df = pd.read_csv(train_csv)
    train_data = {"X": train_df.drop(columns=["target"]), "y": train_df["target"]}
    inference_df = pd.read_csv(inference_csv)
    results = run_pipeline_local(
        config,
        project_root=model_root,
        base_path=model_root / "artifacts",
        train={"train_data": train_data},
        inference={"inference_data": inference_df},
    )
    assert "train" in results
    assert "inference" in results
    print("✓ LocalSparkRunner (direct mode)")


def test_build_package():
    """Test building root.zip for example_model."""
    from mlplatform.spark.packager import build_root_zip

    out = build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=model_root / "dist",
    )
    assert out.exists()
    assert out.suffix == ".zip"
    print(f"✓ build_root_zip -> {out}")


def test_run_spark_step():
    """Test run_spark_step with packages (simulates main.py --packages root.zip)."""
    from mlplatform.spark.config_serializer import write_run_config
    from mlplatform.spark.main import run_spark_step
    from mlplatform.spark.packager import build_root_zip

    from mlplatform.local import load_pipeline_config

    build_root_zip(
        project_root=monorepo_root,
        model_package="example_model",
        output_dir=model_root / "dist",
    )
    root_zip = model_root / "dist" / "root.zip"

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    from mlplatform.config.loader import _env_data_to_config
    from mlplatform.config.schema import RunConfig

    step = next(s for s in config.steps if s.name == "train")
    env_data = step.envs.get(config.env) or step.envs.get("dev") or {}
    env_config = _env_data_to_config(env_data)
    run_config = RunConfig(
        step=step,
        pipeline_name=config.pipeline_name,
        model_name=config.model_name,
        version=config.version,
        feature=config.feature,
        env_config=env_config,
        custom=step.custom,
    )
    config_path = model_root / "dist" / "run_config.json"
    write_run_config(run_config, config_path, base_path=str(model_root / "artifacts"))

    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    train_csv = model_root / "dist" / "train_input.csv"
    pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"]).assign(target=y).to_csv(train_csv, index=False)

    result = run_spark_step(
        str(config_path),
        step_name="train",
        packages=str(root_zip),
        input_path=str(train_csv),
    )
    assert result is not None
    print(f"✓ run_spark_step (train with root.zip): {type(result).__name__}")


def main():
    print("Testing example_model...\n")
    test_load_data_csv()
    test_load_config()
    test_run_pipeline_local()
    test_procedural_prediction_csv()
    test_procedural_prediction_load_data()
    test_local_spark_runner()
    test_build_package()
    test_run_spark_step()
    print("\nAll example_model tests passed.")


if __name__ == "__main__":
    main()
