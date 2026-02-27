"""Configuration loading from YAML files."""

from __future__ import annotations

from pathlib import Path

from mlplatform.config.composer import compose_workflow_dict
from mlplatform.config.schema import ModelConfig, WorkflowConfig


def load_workflow_config(
    dag_path: str | Path,
    config_profile: str | None = None,
    domain: str | None = None,
    runtime_overrides: dict | None = None,
) -> WorkflowConfig:
    """Load a workflow from DAG template format with optional composition overlays."""
    dag_path = Path(dag_path)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    data = compose_workflow_dict(
        dag_path=dag_path,
        config_profile=config_profile,
        domain=domain,
        runtime_overrides=runtime_overrides,
    )
    pipeline_type = data.get("pipeline_type", "training")

    models: list[ModelConfig] = []
    for entry in data.get("models", []):
        platform = entry.get("training_platform") or entry.get("serving_platform") or entry.get("platform") or "VertexAI"
        cloud_cfg = entry.get("cloud") or {}
        models.append(ModelConfig(
            model_name=entry["model_name"],
            module=entry.get("module") or entry.get("entrypoint", ""),
            compute=entry.get("compute") or (cloud_cfg.get("compute", {}) or {}).get("class", "s"),
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
            input_path=entry.get("input_path"),
            output_path=entry.get("output_path"),
            depends_on=entry.get("depends_on") or [],
            cloud=cloud_cfg,
            image=entry.get("image") or cloud_cfg.get("image"),
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
