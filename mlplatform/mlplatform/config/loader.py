"""Configuration loading from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mlplatform.config.schema import ModelConfig, WorkflowConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_workflow_config(dag_path: str | Path) -> WorkflowConfig:
    """Load a workflow from the DAG template format."""
    dag_path = Path(dag_path)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    data = _load_yaml(dag_path)
    pipeline_type = data.get("pipeline_type", "training")

    models: list[ModelConfig] = []
    for entry in data.get("models", []):
        platform = entry.get("training_platform") or entry.get("serving_platform") or "VertexAI"
        models.append(ModelConfig(
            model_name=entry["model_name"],
            module=entry.get("module", ""),
            compute=entry.get("compute", "s"),
            platform=platform,
            optional_configs=entry.get("optional_configs") or {},
            prediction_dataset_name=entry.get("prediction_dataset_name"),
            prediction_table_name=entry.get("prediction_table_name"),
            model_id=entry.get("model_id"),
            model_version=entry.get("model_version", "latest"),
            prediction_output_dataset_table=entry.get("prediction_output_dataset_table"),
            predicted_label_column_name=entry.get("predicted_label_column_name"),
            predicted_timestamp_column_name=entry.get("predicted_timestamp_column_name"),
            predicted_probability_column_name=entry.get("predicted_probability_column_name"),
            unique_identifier_column=entry.get("unique_identifier_column"),
        ))

    return WorkflowConfig(
        workflow_name=data.get("workflow_name", "default_workflow"),
        execution_mode=data.get("execution_mode", "sequential"),
        pipeline_type=pipeline_type,
        feature_name=data.get("feature_name", "default"),
        config_version=data.get("config_version", 2),
        models=models,
        log_level=data.get("log_level", "INFO"),
    )
