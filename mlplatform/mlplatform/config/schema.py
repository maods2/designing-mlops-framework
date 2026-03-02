"""Configuration schema for DAG template format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PipelineType = Literal["training", "prediction"]


@dataclass
class TaskConfig:
    """Configuration for a single task within a pipeline.

    Supports both executable tasks (with module) and orchestration-only tasks.
    """

    task_id: str
    task_type: str
    model_name: str = ""
    module: str = ""
    compute: str = "s"
    platform: str = "VertexAI"
    optional_configs: dict[str, Any] = field(default_factory=dict)
    prediction_dataset_name: str | None = None
    prediction_table_name: str | None = None
    model_id: str | None = None
    model_version: str = "latest"
    prediction_output_dataset_table: str | None = None
    predicted_label_column_name: str | None = None
    predicted_timestamp_column_name: str | None = None
    predicted_probability_column_name: str | None = None
    unique_identifier_column: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    condition_task: dict[str, Any] | None = None
    depends_on: list[dict[str, Any]] | None = None

    def to_model_config(self) -> "ModelConfig":
        """Convert to ModelConfig for downstream consumers (ExecutionContext, invocations)."""
        return ModelConfig(
            model_name=self.model_name or self.task_id,
            module=self.module,
            compute=self.compute,
            platform=self.platform,
            optional_configs=self.optional_configs,
            prediction_dataset_name=self.prediction_dataset_name,
            prediction_table_name=self.prediction_table_name,
            model_id=self.model_id,
            model_version=self.model_version,
            prediction_output_dataset_table=self.prediction_output_dataset_table,
            predicted_label_column_name=self.predicted_label_column_name,
            predicted_timestamp_column_name=self.predicted_timestamp_column_name,
            predicted_probability_column_name=self.predicted_probability_column_name,
            unique_identifier_column=self.unique_identifier_column,
            input_path=self.input_path,
            output_path=self.output_path,
        )


@dataclass
class UnifiedPipelineConfig:
    """Full pipeline configuration from the new flat YAML schema."""

    pipeline_name: str
    pipeline_type: PipelineType
    feature_name: str
    schedule: dict[str, str] = field(default_factory=dict)
    environments: dict[str, bool] = field(default_factory=dict)
    tasks: list[TaskConfig] = field(default_factory=list)
    log_level: str = "INFO"
    config_profiles: list[str] = field(default_factory=list)
    base_path: str | None = None
    artifact_bucket: str | None = None
    artifact_namespace: str = "artifacts"
    env: str = "local"


@dataclass
class ModelConfig:
    """Configuration for a single model within a workflow.

    Unified schema supporting both training and prediction DAG entries.
    """

    model_name: str
    module: str
    compute: str = "s"
    platform: str = "VertexAI"
    optional_configs: dict[str, Any] = field(default_factory=dict)
    prediction_dataset_name: str | None = None
    prediction_table_name: str | None = None
    model_id: str | None = None
    model_version: str = "latest"
    prediction_output_dataset_table: str | None = None
    predicted_label_column_name: str | None = None
    predicted_timestamp_column_name: str | None = None
    predicted_probability_column_name: str | None = None
    unique_identifier_column: str | None = None
    input_path: str | None = None
    output_path: str | None = None


@dataclass
class WorkflowConfig:
    """Full workflow configuration parsed from a DAG YAML template."""

    workflow_name: str
    execution_mode: str
    pipeline_type: str
    feature_name: str
    config_version: int
    models: list[ModelConfig]
    log_level: str = "INFO"
    config_profiles: list[str] = field(default_factory=list)
