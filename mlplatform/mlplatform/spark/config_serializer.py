"""Serialize workflow/runtime config for Spark/Dataproc main.py consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from mlplatform.config.schema import ModelConfig, TaskConfig, UnifiedPipelineConfig
from mlplatform.core.artifact_path_builder import ArtifactPathBuilder


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
    effective_bucket = base_path or pipeline.artifact_bucket or pipeline.base_path or "./artifacts"
    effective_namespace = pipeline.artifact_namespace or "artifacts"
    effective_env = pipeline.env or "local"

    runtime_config: dict[str, Any] = {
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
    }

    env_metadata: dict[str, Any] = {
        "base_path": effective_bucket,
        "profile": profile,
        "execution_mode": "sequential",
        "config_version": 2,
        "log_level": pipeline.log_level,
    }

    builder = ArtifactPathBuilder(
        env=effective_env,
        artifact_bucket=effective_bucket,
        artifact_namespace=effective_namespace,
    )
    paths = builder.build_artifact_paths(
        feature_name=pipeline.feature_name,
        model_name=model.model_name,
        version=version,
        pipeline_type="training",
    )
    runtime_config["artifact_paths"] = {
        "artifact_bucket": paths.artifact_bucket,
        "artifact_base_path": paths.artifact_base_path,
        "artifact_path": paths.artifact_path,
        "model_artifact_dir": paths.model_artifact_dir,
        "metrics_artifact_dir": paths.metrics_artifact_dir,
        "storage_base_path": paths.storage_base_path,
        "metrics_path": paths.metrics_path,
    }
    env_metadata["storage_base_path"] = paths.storage_base_path
    env_metadata["metrics_path"] = paths.metrics_path

    return {
        "runtime_config": runtime_config,
        "environment_metadata": env_metadata,
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
