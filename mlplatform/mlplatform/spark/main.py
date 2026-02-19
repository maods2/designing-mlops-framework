"""Unified Spark entry point for both local and cloud execution.

Usage:
  Local:  python main.py --config <path.json>
  Cloud:  spark-submit main.py --py-files root.zip -- --config <path.json> [--input-path <path>] [--output-path <path>]

The config JSON is produced by config_serializer.write_workflow_config().
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
            from google.cloud import storage as gcs

            parts = config_path[5:].split("/", 1)
            bucket_name, blob_path = parts[0], parts[1]
            client = gcs.Client()
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


def _build_context_from_config(config: dict[str, Any]):
    """Build an ExecutionContext from the serialized run config JSON."""
    from mlplatform.artifacts.local import LocalArtifactStore
    from mlplatform.core.context import ExecutionContext
    from mlplatform.log import get_logger
    from mlplatform.storage.local import LocalFileSystem
    from mlplatform.tracking.local import LocalJsonTracker

    runtime = config.get("runtime_config", {})
    env_meta = config.get("environment_metadata", {})
    base_path = env_meta.get("base_path", "./artifacts")

    return ExecutionContext(
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        feature_name=runtime.get("feature_name", "default"),
        model_name=runtime.get("model_name", "default"),
        version=runtime.get("version", "dev"),
        optional_configs=runtime.get("optional_configs", {}),
        log=get_logger(f"mlplatform.spark.{runtime.get('model_name', 'default')}"),
        _pipeline_type=runtime.get("pipeline_type", ""),
    )


def _resolve_class_from_config(config: dict[str, Any], base_class: type) -> type:
    """Import the module from config and find the first subclass of base_class."""
    runtime = config.get("runtime_config", {})
    module_path = runtime.get("module", "")
    if not module_path:
        raise ValueError("runtime_config must contain 'module'")
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
            return attr
    raise ImportError(f"No {base_class.__name__} subclass found in {module_path}")


def _run_spark_training(config_path: str, packages: str | None = None) -> None:
    """Run training on the Spark driver (in-process)."""
    if packages:
        _add_packages_to_path(packages)

    config = _load_config(config_path)
    from mlplatform.core.trainer import BaseTrainer

    ctx = _build_context_from_config(config)
    trainer_cls = _resolve_class_from_config(config, BaseTrainer)
    trainer = trainer_cls()
    trainer.context = ctx
    ctx.log.info("Starting Spark training")
    trainer.train()
    ctx.log.info("Spark training completed")


def _make_map_in_pandas_fn(config: dict[str, Any]) -> Any:
    """Build mapInPandas function that loads model per partition and runs predict_chunk."""

    def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from mlplatform.core.predictor import BasePredictor

        ctx = _build_context_from_config(config)
        predictor_cls = _resolve_class_from_config(config, BasePredictor)
        predictor = predictor_cls()
        predictor.context = ctx
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
    output_path: str | None = None,
    packages: str | None = None,
) -> None:
    """Run distributed inference via Spark mapInPandas."""
    from pyspark.sql import SparkSession

    if packages:
        _add_packages_to_path(packages)

    config = _load_config(config_path)
    spark = SparkSession.builder.appName("MLPlatform-Spark-Inference").getOrCreate()

    if input_path:
        if input_path.lower().endswith(".parquet"):
            sdf = spark.read.parquet(input_path)
        else:
            sdf = spark.read.option("header", "true").csv(input_path)
    else:
        raise ValueError("--input-path required for inference")

    predict_fn = _make_map_in_pandas_fn(config)

    from pyspark.sql.types import DoubleType, StructField, StructType
    result_schema = StructType(list(sdf.schema.fields) + [StructField("prediction", DoubleType())])
    result = sdf.mapInPandas(predict_fn, schema=result_schema)

    if output_path:
        result.write.mode("overwrite").parquet(output_path)
        print(f"Wrote predictions to {output_path}")
    else:
        result.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPlatform Spark entry point.")
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument("--input-path", help="Input data path (CSV or Parquet)")
    parser.add_argument("--output-path", help="Output path for predictions (Parquet)")
    parser.add_argument("--packages", help="Comma-separated paths to zip packages (e.g. root.zip)")
    args, _ = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)

    config = _load_config(args.config)
    pipeline_type = config.get("runtime_config", {}).get("pipeline_type", "inference")

    try:
        if pipeline_type in ("prediction", "inference"):
            if not args.input_path:
                print("--input-path required for inference", file=sys.stderr)
                return 1
            _run_spark_inference(
                args.config,
                input_path=args.input_path,
                output_path=args.output_path,
                packages=args.packages,
            )
        else:
            _run_spark_training(args.config, packages=args.packages)
        return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
