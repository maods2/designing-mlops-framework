"""Configuration schema for the new DAG template format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    # Prediction-specific fields
    prediction_dataset_name: str | None = None
    prediction_table_name: str | None = None
    model_id: str | None = None
    model_version: str = "latest"
    prediction_output_dataset_table: str | None = None
    predicted_label_column_name: str | None = None
    predicted_timestamp_column_name: str | None = None
    predicted_probability_column_name: str | None = None
    unique_identifier_column: str | None = None


@dataclass
class WorkflowConfig:
    """Full workflow configuration parsed from a DAG YAML template.

    Replaces the old PipelineConfig. Top-level pipeline_type determines
    whether this is a training or prediction workflow.
    """

    workflow_name: str
    execution_mode: str  # sequential, parallel
    pipeline_type: str  # training, prediction
    feature_name: str
    config_version: int
    models: list[ModelConfig]


# --- Legacy schemas kept for backward compatibility during migration ---


@dataclass
class StepConfig:
    """Legacy: Configuration for a single step."""

    name: str
    type: str
    module: str
    class_name: str
    envs: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Legacy: Environment-specific configuration."""

    runner: str
    storage: str
    etb: str
    serving_mode: str = "ProceduralLocal"
    base_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Legacy: Full pipeline configuration."""

    pipeline_name: str
    model_name: str
    version: str
    feature: str
    steps: list[StepConfig]
    env: str


@dataclass
class RunConfig:
    """Legacy: Merged configuration for a single step execution."""

    step: StepConfig
    pipeline_name: str
    model_name: str
    version: str
    feature: str
    env_config: EnvConfig
    custom: dict[str, Any] = field(default_factory=dict)
