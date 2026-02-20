"""Unified Spark entry point for both local and cloud execution.

Usage:
  Local:  python main.py --config <path.json>
  Cloud:  spark-submit main.py --py-files root.zip -- --config <path.json>

The config JSON is produced by config_serializer.write_workflow_config().
This module is kept simple and flexible: it loads zip-packaged DS model code
dynamically and dispatches to either training or prediction via the profile
system. The same entry point works for both local Spark and cloud Spark
(Dataproc / VertexAI).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


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
    """Build an ExecutionContext using the profile system."""
    from mlplatform.core.artifact_registry import ArtifactRegistry
    from mlplatform.core.context import ExecutionContext
    from mlplatform.log import get_logger
    from mlplatform.profiles.registry import get_profile

    runtime = config.get("runtime_config", {})
    env_meta = config.get("environment_metadata", {})
    base_path = env_meta.get("base_path", "./artifacts")
    profile_name = env_meta.get("profile", "local")

    prof = get_profile(profile_name)
    storage = prof.storage_factory(base_path)
    tracker = prof.tracker_factory(base_path)

    feature = runtime.get("feature_name", "default")
    model = runtime.get("model_name", "default")
    ver = runtime.get("version", "dev")

    registry = ArtifactRegistry(
        storage=storage, feature_name=feature, model_name=model, version=ver,
    )
    return ExecutionContext(
        artifacts=registry,
        experiment_tracker=tracker,
        feature_name=feature,
        model_name=model,
        version=ver,
        optional_configs=runtime.get("optional_configs", {}),
        log=get_logger(f"mlplatform.spark.{model}"),
        _pipeline_type=runtime.get("pipeline_type", ""),
    )


def _build_model_cfg_from_config(config: dict[str, Any]):
    """Reconstruct a ModelConfig from the serialized runtime config."""
    from mlplatform.config.schema import ModelConfig

    runtime = config.get("runtime_config", {})
    return ModelConfig(
        model_name=runtime.get("model_name", "default"),
        module=runtime.get("module", ""),
        compute=runtime.get("compute", "s"),
        platform=runtime.get("platform", "VertexAI"),
        optional_configs=runtime.get("optional_configs", {}),
        model_version=runtime.get("model_version", "latest"),
        input_path=runtime.get("input_path"),
        output_path=runtime.get("output_path"),
        prediction_dataset_name=runtime.get("prediction_dataset_name"),
        prediction_table_name=runtime.get("prediction_table_name"),
        prediction_output_dataset_table=runtime.get("prediction_output_dataset_table"),
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


def _run_spark_training(config: dict[str, Any]) -> None:
    """Run training on the Spark driver (in-process)."""
    from mlplatform.core.trainer import BaseTrainer

    ctx = _build_context_from_config(config)
    trainer_cls = _resolve_class_from_config(config, BaseTrainer)
    trainer = trainer_cls()
    trainer.context = ctx
    ctx.log.info("Starting Spark training")
    trainer.train()
    ctx.log.info("Spark training completed")


def _run_spark_inference(config: dict[str, Any]) -> None:
    """Run distributed inference via SparkBatchInvocation."""
    from mlplatform.core.predictor import BasePredictor
    from mlplatform.invocation.spark_batch import SparkBatchInvocation

    ctx = _build_context_from_config(config)
    model_cfg = _build_model_cfg_from_config(config)
    predictor_cls = _resolve_class_from_config(config, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx

    invocation = SparkBatchInvocation()
    invocation.invoke(predictor, ctx, model_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPlatform Spark entry point.")
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument("--packages", help="Comma-separated paths to zip packages (e.g. root.zip)")
    args, _ = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)

    config = _load_config(args.config)
    pipeline_type = config.get("runtime_config", {}).get("pipeline_type", "inference")

    try:
        if pipeline_type in ("prediction", "inference"):
            _run_spark_inference(config)
        else:
            _run_spark_training(config)
        return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
