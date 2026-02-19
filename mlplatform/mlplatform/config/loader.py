"""Configuration loading from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mlplatform.config.schema import EnvConfig, PipelineConfig, StepConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _env_data_to_config(env_data: dict[str, Any]) -> EnvConfig:
    """Build EnvConfig from step's env section. base_path not in config - orchestrator injects."""
    extra = {k: v for k, v in env_data.items() if k not in ("runner", "storage", "etb", "serving_mode", "base_path")}
    return EnvConfig(
        runner=env_data.get("runner", "LocalRunner"),
        storage=env_data.get("storage", "LocalFileSystem"),
        etb=env_data.get("etb", "LocalJsonTracker"),
        serving_mode=env_data.get("serving_mode", "ProceduralLocal"),
        base_path=None,
        extra=extra,
    )


def load_step_config(step_name: str, steps_dir: str | Path) -> StepConfig:
    """Load step configuration from steps directory. Configs (runner, storage, etb) defined per step."""
    path = Path(steps_dir) / f"{step_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Step config not found: {path}")
    data = _load_yaml(path)
    core_keys = {"name", "type", "module", "class", "envs"}
    custom = {k: v for k, v in data.items() if k not in core_keys}
    envs = data.get("envs") or {}
    return StepConfig(
        name=data["name"],
        type=data["type"],
        module=data["module"],
        class_name=data["class"],
        envs=envs,
        custom=custom,
    )


def load_pipeline_config(
    dag_path: str | Path,
    steps_dir: str | Path,
    env: str,
    envs_dir: str | Path | None = None,  # Deprecated: configs now in step YAML
    version: str | None = None,
) -> PipelineConfig:
    """Load and merge DAG, step, and environment configuration."""
    dag_path = Path(dag_path)
    steps_dir = Path(steps_dir)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    dag_data = _load_yaml(dag_path)
    pipeline_data = dag_data.get("pipeline", dag_data)
    pipeline_name = pipeline_data.get("name", "default_pipeline")
    model_name = pipeline_data.get("model_name", "default_model")
    # Version: timestamp + short id when not specified in config (sortable, unique)
    version = version or pipeline_data.get("version")
    if not version or str(version).lower() in ("null", "none", ""):
        import uuid
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        version = f"{ts}_{short_id}"
    # Feature: domain/use-case level, distinct from model (avoids feature/model duplication)
    feature = pipeline_data.get("feature") or pipeline_name

    steps: list[StepConfig] = []
    for step_entry in pipeline_data.get("steps", []):
        if isinstance(step_entry, str):
            step_name = step_entry
            dag_overrides = {}
        else:
            step_name = step_entry.get("name", "unknown")
            dag_overrides = {k: v for k, v in step_entry.items() if k != "custom"}
            dag_overrides.pop("name", None)
        step_config = load_step_config(step_name, steps_dir)
        if dag_overrides:
            for k, v in dag_overrides.items():
                if k in ("type", "module", "class"):
                    setattr(step_config, "class_name" if k == "class" else k, v)
                else:
                    step_config.custom[k] = v
        if isinstance(step_entry, dict) and "custom" in step_entry:
            step_config.custom.update(step_entry["custom"])
        steps.append(step_config)

    return PipelineConfig(
        pipeline_name=pipeline_name,
        model_name=model_name,
        version=version,
        feature=feature,
        steps=steps,
        env=env,
    )
