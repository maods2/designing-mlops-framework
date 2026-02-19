"""Serialize workflow/runtime config for Spark/Dataproc main.py consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform.config.schema import ModelConfig, WorkflowConfig


def workflow_config_to_dict(
    workflow: WorkflowConfig,
    model: ModelConfig,
    base_path: str = "./artifacts",
    version: str = "dev",
) -> dict[str, Any]:
    """Serialize workflow + model config to JSON-serializable dict for cloud main.py."""
    return {
        "runtime_config": {
            "workflow_name": workflow.workflow_name,
            "pipeline_type": workflow.pipeline_type,
            "feature_name": workflow.feature_name,
            "model_name": model.model_name,
            "module": model.module,
            "class_name": _resolve_class_name(model.module),
            "version": version,
            "compute": model.compute,
            "platform": model.platform,
            "optional_configs": model.optional_configs,
            "model_version": model.model_version,
        },
        "environment_metadata": {
            "base_path": base_path,
            "execution_mode": workflow.execution_mode,
            "config_version": workflow.config_version,
        },
    }


def _resolve_class_name(module_path: str) -> str:
    """Resolve the expected class name from a module path by convention.

    Convention: the module's primary class is discovered at runtime by main.py
    by inspecting the module for BaseTrainer/BasePredictor subclasses.
    Falls back to module filename CamelCased.
    """
    parts = module_path.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].title().replace("_", "")
    return module_path.title().replace("_", "")


def write_workflow_config(
    workflow: WorkflowConfig,
    model: ModelConfig,
    path: str | Path,
    base_path: str = "./artifacts",
    version: str = "dev",
) -> Path:
    """Write workflow config to JSON file for cloud submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            workflow_config_to_dict(workflow, model, base_path=base_path, version=version),
            f,
            indent=2,
        )
    return path
