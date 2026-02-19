"""Configuration schema for DAG template format."""

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
    """Full workflow configuration parsed from a DAG YAML template."""

    workflow_name: str
    execution_mode: str
    pipeline_type: str
    feature_name: str
    config_version: int
    models: list[ModelConfig]
    log_level: str = "INFO"
