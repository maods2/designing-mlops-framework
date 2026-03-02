"""Unified Spark entry point for both local and cloud execution.

Usage:
  Local (no zip):     python main.py --config <path.json>
  Local (with zip):   python main.py --config <path.json> --packages dist/root.zip
  Cloud spark-submit: spark-submit main.py --py-files dist/root.zip -- --config <path.json>

Path resolution:
  if --packages provided: add each zip to sys.path (cloud-like mode)
  else: add project_root and mlplatform to sys.path (local dev mode)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_local_paths(project_root: Path | None = None) -> None:
    """Add project_root and mlplatform to sys.path for local dev testing."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    for p in [str(project_root), str(project_root / "mlplatform")]:
        if p not in sys.path:
            sys.path.insert(0, p)


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
    profile_name = env_meta.get("profile", "local")

    # Use standardized paths when serialized by config_serializer
    storage_base = env_meta.get("storage_base_path")
    if storage_base is None:
        storage_base = env_meta.get("base_path", "./artifacts")
    metrics_path = env_meta.get("metrics_path")

    prof = get_profile(profile_name)
    storage = prof.storage_factory(storage_base)
    tracker = prof.tracker_factory(storage_base, metrics_path=metrics_path)

    feature = runtime.get("feature_name", "default")
    model = runtime.get("model_name", "default")
    ver = runtime.get("version", "dev")

    artifact_paths = runtime.get("artifact_paths", {})
    registry_kwargs: dict[str, Any] = {}
    if artifact_paths:
        registry_kwargs = {
            "storage_base_path": artifact_paths.get("storage_base_path"),
            "artifact_path": artifact_paths.get("artifact_path"),
            "model_artifact_dir": artifact_paths.get("model_artifact_dir"),
            "metrics_artifact_dir": artifact_paths.get("metrics_artifact_dir"),
        }

    registry = ArtifactRegistry(
        storage=storage,
        feature_name=feature,
        model_name=model,
        version=ver,
        **registry_kwargs,
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
        commit_hash=runtime.get("commit_hash"),
    )


def _build_model_cfg_from_config(
    config: dict[str, Any],
    input_path: str | None = None,
    output_path: str | None = None,
):
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
        input_path=input_path or runtime.get("input_path"),
        output_path=output_path or runtime.get("output_path"),
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


def _log_framework_params(ctx: Any, config: dict[str, Any]) -> None:
    """Log framework-level parameters for reproducibility tracking."""
    env_meta = config.get("environment_metadata", {})
    params: dict[str, Any] = {
        "mlplatform.profile": env_meta.get("profile", "unknown"),
        "mlplatform.version": ctx.version,
        "mlplatform.pipeline_type": ctx._pipeline_type,
    }
    if ctx.commit_hash:
        params["mlplatform.commit_hash"] = ctx.commit_hash
    ctx.log_params(params)


def _run_spark_training(config: dict[str, Any]) -> None:
    """Run training on the Spark driver (in-process)."""
    from mlplatform.core.trainer import BaseTrainer

    ctx = _build_context_from_config(config)
    _log_framework_params(ctx, config)
    trainer_cls = _resolve_class_from_config(config, BaseTrainer)
    trainer = trainer_cls()
    trainer.context = ctx
    ctx.log.info("Starting Spark training")
    trainer.train()
    ctx.log.info("Spark training completed")


def _run_spark_inference(
    config: dict[str, Any],
    input_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run distributed inference via SparkBatchInvocation.

    Args:
        config: Serialized runtime config dict.
        input_path: Override input path from CLI (optional).
        output_path: Override output path from CLI (optional).
    """
    from mlplatform.core.predictor import BasePredictor
    from mlplatform.invocation.spark_batch import SparkBatchInvocation

    ctx = _build_context_from_config(config)
    _log_framework_params(ctx, config)
    model_cfg = _build_model_cfg_from_config(config, input_path=input_path, output_path=output_path)
    predictor_cls = _resolve_class_from_config(config, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx

    invocation = SparkBatchInvocation()
    invocation.invoke(predictor, ctx, model_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPlatform Spark entry point.")
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument(
        "--input-path",
        help="Override input path from config (file path or gs:// URI)",
    )
    parser.add_argument(
        "--output-path",
        help="Override output path from config (file path or gs:// URI)",
    )
    parser.add_argument(
        "--packages",
        help="Comma-separated paths to zip packages (e.g. dist/root.zip). "
             "When provided, zips are added to sys.path. "
             "When omitted, project_root is added for local dev mode.",
    )
    parser.add_argument(
        "--project-root",
        help="Project root for local path bootstrap (default: inferred). "
             "Only used when --packages is not provided.",
    )
    args, _ = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)
    else:
        project_root = Path(args.project_root).resolve() if args.project_root else None
        _bootstrap_local_paths(project_root)

    config = _load_config(args.config)
    pipeline_type = config.get("runtime_config", {}).get("pipeline_type", "inference")

    try:
        if pipeline_type in ("prediction", "inference"):
            _run_spark_inference(
                config,
                input_path=args.input_path,
                output_path=args.output_path,
            )
        else:
            _run_spark_training(config)
        return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
