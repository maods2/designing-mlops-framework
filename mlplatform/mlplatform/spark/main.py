"""
Entry point for Spark/Dataproc cluster execution (cloud only).

Invocation format (Dataproc/Spark):
  spark-submit main.py --py-files root.zip -- --config <path> [--input-path <path>] [--output-path <path>]

main.py is an auxiliary job used ONLY during cloud job submission. It drives distributed
prediction via mapInPandas, passing the user's predictor class to each partition.
For training, it runs the trainer in-process on the driver.

For local execution: use LocalJobRunner or LocalSparkJobRunner with direct=True.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def _add_packages_to_path(packages: str) -> None:
    """Add zip packages to sys.path so model code can be imported."""
    for p in packages.split(","):
        p = p.strip()
        if not p:
            continue
        path = Path(p)
        if path.suffix == ".zip" and path.exists():
            sys.path.insert(0, str(path.resolve()))


def _load_config(config_path: str) -> dict[str, Any]:
    """Load run config from local file or GCS."""
    if config_path.startswith("gs://"):
        try:
            from google.cloud import storage

            parts = config_path[5:].split("/", 1)
            bucket_name, blob_path = parts[0], parts[1]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            return json.loads(content)
        except ImportError:
            raise RuntimeError(
                "google-cloud-storage required for GCS config. pip install google-cloud-storage"
            )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return json.load(f)


def _resolve_storage(config: dict[str, Any]):
    """Instantiate storage from config."""
    from mlplatform.storage.local import LocalFileSystem

    env_meta = config.get("environment_metadata", {})
    base_path = env_meta.get("base_path", "./artifacts")
    return LocalFileSystem(base_path=base_path)


def _make_map_in_pandas_fn(
    predictor_module: str,
    predictor_class: str,
    config: dict[str, Any],
) -> Any:
    """Build mapInPandas function that loads model per partition and runs predict_chunk."""

    def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        mod = importlib.import_module(predictor_module)
        predictor_cls = getattr(mod, predictor_class)
        storage = _resolve_storage(config)

        runtime = config.get("runtime_config", {})
        feature = runtime.get("feature_name", "default")
        model_name = runtime.get("model_name", "default")
        version = runtime.get("version", "dev")
        model_path = f"{feature}/{model_name}/{version}/model.pkl"

        predictor = predictor_cls()
        predictor.context = None
        from mlplatform.core.context import ExecutionContext
        from mlplatform.artifacts.local import LocalArtifactStore
        predictor.context = ExecutionContext(
            storage=storage,
            artifact_store=LocalArtifactStore(base_path=str(storage.base_path)),
            experiment_tracker=None,
            invocation_strategy=None,
            runtime_config=runtime,
            environment_metadata=config.get("environment_metadata", {}),
        )
        predictor.load_model()

        for batch in iterator:
            result = predictor.predict_chunk(batch)
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"prediction": result})
            yield result

    return predict_partition


def _run_spark_inference(
    config_path: str,
    input_path: str | None = None,
    input_source: dict[str, Any] | None = None,
    output_path: str | None = None,
    packages: str | None = None,
) -> None:
    """Run distributed inference via Spark mapInPandas."""
    from pyspark.sql import SparkSession

    if packages:
        _add_packages_to_path(packages)

    config = _load_config(config_path)
    runtime = config.get("runtime_config", {})
    predictor_module = runtime.get("module", "")
    predictor_class = runtime.get("class_name", "")

    if not predictor_module or not predictor_class:
        raise ValueError("runtime_config must contain 'module' and 'class_name'")

    spark = SparkSession.builder.appName("MLPlatform-Spark-Inference").getOrCreate()

    if input_source:
        from mlplatform.data import load_data
        sdf = load_data(input_source, format="spark", spark=spark)
    elif input_path:
        if input_path.lower().endswith(".parquet"):
            sdf = spark.read.parquet(input_path)
        else:
            sdf = spark.read.option("header", "true").csv(input_path)
    else:
        raise ValueError("Either input_path or input_source required for inference")

    predict_fn = _make_map_in_pandas_fn(predictor_module, predictor_class, config)

    from pyspark.sql.types import DoubleType
    result_schema = sdf.schema.add("prediction", DoubleType())
    result = sdf.mapInPandas(predict_fn, schema=result_schema)

    if output_path:
        result.write.mode("overwrite").parquet(output_path)
        print(f"Wrote predictions to {output_path}")
    else:
        result.show()


def _run_spark_training(
    config_path: str,
    packages: str | None = None,
) -> None:
    """Run training on the Spark driver (in-process)."""
    if packages:
        _add_packages_to_path(packages)

    config = _load_config(config_path)
    runtime = config.get("runtime_config", {})
    trainer_module = runtime.get("module", "")
    trainer_class = runtime.get("class_name", "")

    if not trainer_module or not trainer_class:
        raise ValueError("runtime_config must contain 'module' and 'class_name'")

    mod = importlib.import_module(trainer_module)
    trainer_cls = getattr(mod, trainer_class)

    storage = _resolve_storage(config)
    from mlplatform.artifacts.local import LocalArtifactStore
    from mlplatform.etb.local_json import LocalJsonTracker
    from mlplatform.core.context import ExecutionContext

    env_meta = config.get("environment_metadata", {})
    base_path = env_meta.get("base_path", "./artifacts")

    context = ExecutionContext(
        storage=storage,
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        invocation_strategy=None,
        runtime_config=runtime,
        environment_metadata=env_meta,
    )

    trainer = trainer_cls()
    trainer.context = context
    context.trainer = trainer
    trainer.train()
    print("Training completed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLPlatform Spark entry point (cloud only)."
    )
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument("--input-path", help="Input data path (CSV or Parquet)")
    parser.add_argument(
        "--input-source",
        help='Data source config JSON, e.g. \'{"type":"bigquery","table":"project.dataset.table"}\'',
    )
    parser.add_argument("--output-path", help="Output path for predictions (Parquet)")
    parser.add_argument("--packages", help="Comma-separated paths to zip packages (e.g. root.zip)")
    args, _ = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)

    config = _load_config(args.config)
    pipeline_type = config.get("runtime_config", {}).get("pipeline_type", "inference")

    try:
        if pipeline_type in ("prediction", "inference"):
            input_source = None
            if getattr(args, "input_source", None):
                input_source = json.loads(args.input_source)
            if not args.input_path and not input_source:
                print("--input-path or --input-source required for inference", file=sys.stderr)
                return 1
            _run_spark_inference(
                args.config,
                input_path=args.input_path,
                input_source=input_source,
                output_path=args.output_path,
                packages=args.packages,
            )
            return 0
        else:
            _run_spark_training(args.config, packages=args.packages)
            return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
