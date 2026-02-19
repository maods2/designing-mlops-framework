#!/usr/bin/env python
"""Test Dataproc/Spark execution format locally (no Spark cluster required)."""

import sys
from pathlib import Path

monorepo_root = Path(__file__).resolve().parent.parent
model_root = monorepo_root / "template_model"
sys.path.insert(0, str(monorepo_root))


def test_build_package():
    """Test building root.zip under template_model/dist."""
    from mlplatform.spark.packager import build_root_zip

    out = build_root_zip(
        project_root=monorepo_root,
        model_package="template_model",
        output_dir=model_root / "dist",
    )
    assert out.exists()
    assert out.suffix == ".zip"
    print(f"✓ build_root_zip -> {out}")


def test_run_spark_main_direct():
    """Test run_spark_step with packages (simulates main.py --packages root.zip)."""
    from mlplatform.spark.config_serializer import write_run_config
    from mlplatform.spark.main import run_spark_step
    from mlplatform.spark.packager import build_root_zip

    from mlplatform.local import load_pipeline_config

    # Build package
    build_root_zip(
        project_root=monorepo_root,
        model_package="template_model",
        output_dir=model_root / "dist",
    )
    root_zip = model_root / "dist" / "root.zip"

    # Load config and write run_config (env from step YAML)
    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="dev",
    )
    from mlplatform.config.loader import _env_data_to_config
    from mlplatform.config.schema import RunConfig

    step = config.steps[0]
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

    # Run train step via run_spark_step (with packages)
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=30, n_features=5, random_state=42)
    train_csv = model_root / "dist" / "train_input.csv"
    pd.DataFrame(X).assign(target=y).to_csv(train_csv, index=False)

    result = run_spark_step(
        str(config_path),
        step_name="train",
        packages=str(root_zip),
        input_path=str(train_csv),
    )
    assert result is not None
    print(f"✓ run_spark_step (train with root.zip): {type(result).__name__}")


def test_local_spark_runner():
    """Test LocalSparkRunner in direct mode (in-process, no packaging)."""
    from mlplatform.local import load_pipeline_config, run_pipeline_local

    import pandas as pd
    from sklearn.datasets import make_classification

    config = load_pipeline_config(
        dag_path=model_root / "pipeline/dags/train_infer.yaml",
        steps_dir=model_root / "pipeline/steps",
        env="local_spark",
    )
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    results = run_pipeline_local(
        config,
        project_root=model_root,
        train={"train_data": {"X": pd.DataFrame(X), "y": pd.Series(y)}},
        inference={"inference_data": pd.DataFrame(X[:5])},
    )
    assert "train" in results
    assert "inference" in results
    print("✓ LocalSparkRunner (direct mode)")


def main():
    print("Testing Spark/Dataproc local execution...\n")
    test_build_package()
    test_run_spark_main_direct()
    test_local_spark_runner()
    print("\nAll Spark local tests passed.")


if __name__ == "__main__":
    main()
