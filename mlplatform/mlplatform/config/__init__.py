"""Configuration loading and schema."""

from mlplatform.config.factory import (
    ConfigLoaderFactory,
    PipelineConfigLoader,
)
from mlplatform.config.schema import (
    ModelConfig,
    TaskConfig,
    UnifiedPipelineConfig,
    WorkflowConfig,
)

# Primary API: factory-based loader for new schema
load_pipeline_config = ConfigLoaderFactory.load_pipeline_config

__all__ = [
    "ConfigLoaderFactory",
    "ModelConfig",
    "PipelineConfigLoader",
    "TaskConfig",
    "UnifiedPipelineConfig",
    "WorkflowConfig",
    "load_pipeline_config",
]
