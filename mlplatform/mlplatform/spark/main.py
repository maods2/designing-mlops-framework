"""
Entry point for Spark/Dataproc cluster execution (cloud only).

Invocation format (Dataproc/Spark):
  spark-submit main.py --py-files root.zip -- --config <path> --input-path <path> [--output-path <path>]

main.py is an auxiliary job used ONLY during cloud job submission. It drives distributed
prediction via mapInPandas, passing the user's predictor class to each partition.

For local execution: use LocalRunner or LocalSparkRunner with direct=True. Do NOT run
main.py locally - it requires Spark and is intended for cluster submission only.
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
    from mlplatform.config.registry import get_storage

    env = config.get("env_config", {})
    return get_storage(
        env.get("storage", "LocalFileSystem"),
        base_path=env.get("base_path", "./artifacts"),
    )


def _make_map_in_pandas_fn(
    predictor_cls: type,
    config: dict[str, Any],
) -> Any:
    """
    Build mapInPandas function that loads model per partition and runs predict_chunk.

    The predictor class must implement load_model(storage, path) and predict_chunk(data).
    predict_chunk receives a pandas DataFrame and must return a pandas DataFrame
    (with predictions added).
    """

    def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        storage = _resolve_storage(config)
        feature = config.get("feature", config.get("model_name", "default"))
        model_name = config.get("model_name", "default")
        version = config.get("version", "dev")
        model_path = f"{feature}/{model_name}/{version}/model.pkl"

        predictor = predictor_cls()
        model = predictor.load_model(storage, model_path)

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
    """
    Run distributed inference via Spark mapInPandas.

    Reads input from input_path (CSV/Parquet) or input_source (e.g. BigQuery),
    applies predictor via mapInPandas, writes to output_path if provided.
    """
    from pyspark.sql import SparkSession

    if packages:
        _add_packages_to_path(packages)

    config = _load_config(config_path)
    step_config = config.get("step", {})
    mod = importlib.import_module(step_config["module"])
    predictor_cls = getattr(mod, step_config["class_name"])

    if not hasattr(predictor_cls, "load_model") or not hasattr(predictor_cls, "predict_chunk"):
        raise TypeError(
            f"Predictor {predictor_cls.__name__} must implement load_model and predict_chunk "
            "for Spark mapInPandas. See BasePredictor."
        )

    spark = SparkSession.builder.appName("MLPlatform-Spark-Inference").getOrCreate()

    # Read input - from path or data source config
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

    predict_fn = _make_map_in_pandas_fn(predictor_cls, config)
    # Schema: input columns + prediction (predictor returns input.assign(prediction=...))
    from pyspark.sql.types import DoubleType

    result_schema = sdf.schema.add("prediction", DoubleType())

    result = sdf.mapInPandas(predict_fn, schema=result_schema)

    if output_path:
        result.write.mode("overwrite").parquet(output_path)
        print(f"Wrote predictions to {output_path}")
    else:
        result.show()


def run_spark_step(
    config_path: str,
    step_name: str | None = None,
    packages: str | None = None,
    input_path: str | None = None,
    input_source: dict[str, Any] | None = None,
    **step_kwargs: Any,
) -> Any:
    """
    Run a single step from serialized config. Used for LOCAL execution only.

    For inference: when running locally (LocalSparkRunner direct=True), this runs
    the step in-process without Spark. Do NOT use for cloud - cloud uses
    _run_spark_inference with mapInPandas instead.
    """
    import importlib

    if packages:
        _add_packages_to_path(packages)
    config = _load_config(config_path)
    if step_name:
        config["step"] = {**config.get("step", {}), "name": step_name}

    step_type = config.get("step", {}).get("type", "")
    if input_source and step_type == "inference":
        from mlplatform.data import load_data

        step_kwargs["inference_data"] = load_data(input_source, format="pandas")
    elif input_path:
        if step_type == "train":
            step_kwargs["train_data"] = _load_train_data_from_path(input_path)
        elif step_type == "inference":
            if input_path.lower().endswith(".parquet"):
                step_kwargs["inference_data"] = pd.read_parquet(input_path)
            else:
                step_kwargs["inference_data"] = pd.read_csv(input_path)

    context = _build_context_from_config(config)
    step_config = context.run_config.step
    mod = importlib.import_module(step_config.module)
    cls = getattr(mod, step_config.class_name)
    step = cls()
    step._context = context
    return step.run(context, **step_kwargs)


def _build_context_from_config(config: dict[str, Any]):
    """Build ExecutionContext from serialized config."""
    from mlplatform.config.schema import EnvConfig, RunConfig, StepConfig
    from mlplatform.core.context import ExecutionContext
    from mlplatform.runners.local import LocalRunner

    step_data = config.get("step", {})
    step_config = StepConfig(
        name=step_data.get("name", "unknown"),
        type=step_data.get("type", "train"),
        module=step_data.get("module", ""),
        class_name=step_data.get("class", step_data.get("class_name", "")),
        custom=step_data.get("custom", {}),
    )
    env_data = config.get("env_config", {})
    env_config = EnvConfig(
        runner=env_data.get("runner", "LocalRunner"),
        storage=env_data.get("storage", "LocalFileSystem"),
        etb=env_data.get("etb", "LocalJsonTracker"),
        serving_mode=env_data.get("serving_mode", "ProceduralLocal"),
        base_path=env_data.get("base_path", "./artifacts"),
        extra=env_data.get("extra", {}),
    )
    run_config = RunConfig(
        step=step_config,
        pipeline_name=config.get("pipeline_name", "default"),
        model_name=config.get("model_name", "default"),
        version=config.get("version", "dev"),
        feature=config.get("feature", config.get("model_name", "default")),
        env_config=env_config,
        custom=step_config.custom,
    )
    from mlplatform.spark.main import _resolve_etb, _resolve_storage

    storage = _resolve_storage(config)
    etb = _resolve_etb(config)
    return ExecutionContext(
        storage=storage,
        etb=etb,
        runner=LocalRunner(),
        run_config=run_config,
        feature=run_config.feature,
        model_name=run_config.model_name,
        version=run_config.version,
        step_name=run_config.step.name,
        custom=run_config.custom,
    )


def _load_train_data_from_path(path: str):
    """Load train data from CSV or Parquet for train step."""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"Data must have 'target' column: {path}")
    return {"X": df.drop(columns=["target"]), "y": df["target"]}


def _resolve_etb(config: dict[str, Any]):
    """Instantiate ETB from config."""
    from mlplatform.config.registry import get_etb

    env = config.get("env_config", {})
    return get_etb(
        env.get("etb", "LocalJsonTracker"),
        base_path=env.get("base_path", "./artifacts"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLPlatform Spark entry point (cloud only). Uses mapInPandas for distributed inference."
    )
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument(
        "--input-path",
        help="Input data path (CSV or Parquet, local or gs://). Required for inference if --input-source not set.",
    )
    parser.add_argument(
        "--input-source",
        help='Data source config JSON, e.g. \'{"type":"bigquery","table":"project.dataset.table"}\'',
    )
    parser.add_argument("--output-path", help="Output path for predictions (Parquet)")
    parser.add_argument(
        "--packages",
        help="Comma-separated paths to zip packages (e.g. root.zip) - for local Spark testing",
    )
    parser.add_argument("--step-name", help="Step name override (ignored for inference)")
    args, unknown = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)

    config = _load_config(args.config)
    step_type = config.get("step", {}).get("type", "inference")

    try:
        if step_type == "inference":
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
            # Train: run in-process (driver) - for small datasets
            result = run_spark_step(
                args.config,
                step_name=args.step_name,
                packages=args.packages,
                input_path=args.input_path,
            )
            print(f"Step completed: {result}")
            return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
