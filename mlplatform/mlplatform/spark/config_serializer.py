"""Serialize workflow/runtime config for Spark/Dataproc main.py consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from mlplatform.config.schema import ModelConfig, TaskConfig, UnifiedPipelineConfig


def workflow_config_to_dict(
    pipeline: UnifiedPipelineConfig,
    task: Union[TaskConfig, ModelConfig],
    base_path: str = "./artifacts",
    version: str = "dev",
    profile: str = "local",
    commit_hash: str | None = None,
) -> dict[str, Any]:
    """Serialize pipeline + task config to JSON-serializable dict for cloud main.py."""
    model = task.to_model_config() if isinstance(task, TaskConfig) else task
    return {
        "runtime_config": {
            "workflow_name": pipeline.pipeline_name,
            "pipeline_type": pipeline.pipeline_type,
            "feature_name": pipeline.feature_name,
            "model_name": model.model_name,
            "module": model.module,
            "version": version,
            "compute": model.compute,
            "platform": model.platform,
            "optional_configs": model.optional_configs,
            "model_version": model.model_version,
            "input_path": model.input_path,
            "output_path": model.output_path,
            "prediction_dataset_name": model.prediction_dataset_name,
            "prediction_table_name": model.prediction_table_name,
            "prediction_output_dataset_table": model.prediction_output_dataset_table,
            "commit_hash": commit_hash,
        },
        "environment_metadata": {
            "base_path": base_path,
            "profile": profile,
            "execution_mode": "sequential",
            "config_version": 2,
            "log_level": pipeline.log_level,
        },
    }


def write_workflow_config(
    pipeline: UnifiedPipelineConfig,
    task: Union[TaskConfig, ModelConfig],
    path: str | Path,
    base_path: str = "./artifacts",
    version: str = "dev",
    profile: str = "local",
    commit_hash: str | None = None,
) -> Path:
    """Write workflow config to JSON file for cloud submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            workflow_config_to_dict(
                pipeline, task, base_path=base_path, version=version,
                profile=profile, commit_hash=commit_hash,
            ),
            f,
            indent=2,
        )
    return path
