"""Configuration schemas, Pydantic models, and YAML loading.

Public API
----------
The primary interface for users of ``mlplatform[config]``:

* :class:`TrainingConfig` — Pydantic model for a single training job.
* :class:`PredictionConfig` — Pydantic model for a single inference job.
* :class:`PipelineConfig` — Pydantic model for a full workflow; supports
  ``PipelineConfig.from_yaml(path)`` to load from a DAG YAML file.

Underlying dataclass-based types (kept for framework internals and backward
compatibility):

* :class:`ModelConfig` — dataclass used internally by the YAML loader.
* :class:`WorkflowConfig` — dataclass representing the raw loaded workflow.
* :func:`load_workflow_config` — low-level YAML loader (used by
  :meth:`PipelineConfig.from_yaml`).

Install
-------
    pip install mlplatform[config]
"""

from mlplatform.config.loader import load_workflow_config
from mlplatform.config.models import PipelineConfig, PredictionConfig, TrainingConfig
from mlplatform.config.schema import ModelConfig, WorkflowConfig

__all__ = [
    # ── Pydantic models (public API) ──────────────────────────────────────────
    "TrainingConfig",
    "PredictionConfig",
    "PipelineConfig",
    # ── Underlying dataclass types (framework internals / advanced use) ────────
    "ModelConfig",
    "WorkflowConfig",
    "load_workflow_config",
]
