"""Configuration schema and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepConfig:
    """Configuration for a single step (from step YAML + DAG step entry).
    envs: per-environment config (dev, qa, prod) - runner, storage, etb, serving_mode.
    """

    name: str
    type: str
    module: str
    class_name: str
    envs: dict[str, dict[str, Any]] = field(default_factory=dict)  # env -> {runner, storage, etb, ...}
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Environment-specific configuration (dev, qa, prod).
    base_path is NOT in config - orchestrator injects it (bucket or root folder per storage type).
    """

    runner: str
    storage: str
    etb: str
    serving_mode: str = "ProceduralLocal"
    base_path: str | None = None  # Injected by orchestrator, not from YAML
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Full pipeline configuration (DAG + steps). Env config resolved per-step from step.envs."""

    pipeline_name: str
    model_name: str
    version: str
    feature: str
    steps: list[StepConfig]
    env: str  # dev, qa, prod - used to resolve step.envs[env]


@dataclass
class RunConfig:
    """Merged configuration for a single step execution."""

    step: StepConfig
    pipeline_name: str
    model_name: str
    version: str
    feature: str
    env_config: EnvConfig
    custom: dict[str, Any] = field(default_factory=dict)
