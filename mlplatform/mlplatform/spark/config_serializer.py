"""Serialize RunConfig for Spark/Dataproc main.py consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform.config.schema import EnvConfig, RunConfig, StepConfig


def run_config_to_dict(
    run_config: RunConfig,
    base_path: str | None = None,
) -> dict[str, Any]:
    """Serialize RunConfig to JSON-serializable dict.
    base_path: Injected by orchestrator (bucket or root folder). Required for storage.
    """
    return {
        "step": {
            "name": run_config.step.name,
            "type": run_config.step.type,
            "module": run_config.step.module,
            "class": run_config.step.class_name,
            "class_name": run_config.step.class_name,
            "custom": run_config.step.custom,
        },
        "pipeline_name": run_config.pipeline_name,
        "model_name": run_config.model_name,
        "version": run_config.version,
        "feature": run_config.feature,
        "env_config": {
            "runner": run_config.env_config.runner,
            "storage": run_config.env_config.storage,
            "etb": run_config.env_config.etb,
            "serving_mode": run_config.env_config.serving_mode,
            "base_path": base_path or run_config.env_config.base_path or "./artifacts",
            "extra": run_config.env_config.extra,
        },
        "custom": run_config.custom,
    }


def write_run_config(
    run_config: RunConfig,
    path: str | Path,
    base_path: str | None = None,
) -> Path:
    """Write RunConfig to JSON file. base_path injected by orchestrator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run_config_to_dict(run_config, base_path=base_path), f, indent=2)
    return path
