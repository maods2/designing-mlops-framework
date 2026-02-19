"""
Entry point for Spark/Dataproc cluster execution.

Invocation format (Dataproc/Spark):
  spark-submit main.py --py-files root.zip -- --config <path> [--step-name <name>]

Invocation format (local testing - no Spark):
  python main.py --config <path> [--packages root.zip] [--step-name <name>]

Config path can be local (./config.json) or GCS (gs://bucket/path/config.json).
"""

from __future__ import annotations

import argparse
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
            # Insert at beginning so model package takes precedence
            sys.path.insert(0, str(path.resolve()))


def _load_config(config_path: str) -> dict[str, Any]:
    """Load run config from local file or GCS."""
    if config_path.startswith("gs://"):
        try:
            from google.cloud import storage

            parts = config_path[5:].split("/", 1)  # gs://bucket/path
            bucket_name, blob_path = parts[0], parts[1]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            return json.loads(content)
        except ImportError:
            raise RuntimeError("google-cloud-storage required for GCS config. pip install google-cloud-storage")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return json.load(f)


def _resolve_storage(config: dict[str, Any]):
    """Instantiate storage from config."""
    from mlplatform.config.registry import get_storage

    env = config.get("env_config", {})
    return get_storage(env.get("storage", "LocalFileSystem"), base_path=env.get("base_path", "./artifacts"))


def _resolve_etb(config: dict[str, Any]):
    """Instantiate ETB from config."""
    from mlplatform.config.registry import get_etb

    env = config.get("env_config", {})
    return get_etb(env.get("etb", "LocalJsonTracker"), base_path=env.get("base_path", "./artifacts"))


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
    """Load train data from CSV for train step."""
    import pandas as pd

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"CSV must have 'target' column: {path}")
    return {"X": df.drop(columns=["target"]), "y": df["target"]}


def run_spark_step(
    config_path: str,
    step_name: str | None = None,
    packages: str | None = None,
    input_path: str | None = None,
    **step_kwargs: Any,
) -> Any:
    """
    Run a single step from serialized config. Used by main.py entry point.

    Args:
        config_path: Path to run config JSON (local or gs://)
        step_name: Optional step name override
        packages: Optional path to root.zip (adds to sys.path before import)
        input_path: Optional path to input CSV (for train step: train_data)
        **step_kwargs: Passed to step.run()

    Returns:
        Step result
    """
    import importlib

    if packages:
        _add_packages_to_path(packages)
    config = _load_config(config_path)
    if step_name:
        config["step"] = {**config.get("step", {}), "name": step_name}

    if input_path:
        step_type = config.get("step", {}).get("type", "")
        if step_type == "train":
            step_kwargs["train_data"] = _load_train_data_from_path(input_path)
        elif step_type == "inference":
            import pandas as pd

            step_kwargs["inference_data"] = pd.read_csv(input_path)

    context = _build_context_from_config(config)
    step_config = context.run_config.step
    mod = importlib.import_module(step_config.module)
    cls = getattr(mod, step_config.class_name)
    step = cls()
    step._context = context
    return step.run(context, **step_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPlatform Spark step entry point")
    parser.add_argument("--config", required=True, help="Path to run config JSON (local or gs://)")
    parser.add_argument("--packages", help="Comma-separated paths to zip packages (e.g. root.zip) for local run")
    parser.add_argument("--step-name", help="Step name override")
    parser.add_argument("--input-path", help="Input data path (e.g. GCS path for Spark)")
    parser.add_argument("--output-path", help="Output path for results")
    args, unknown = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)

    step_kwargs = {}
    if args.output_path:
        step_kwargs["output_path"] = args.output_path

    try:
        result = run_spark_step(
            args.config,
            step_name=args.step_name,
            packages=args.packages,
            input_path=args.input_path,
            **step_kwargs,
        )
        print(f"Step completed: {result}")
        return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
