"""
MLOps Framework - A minimal but extensible framework for ML model development.

This framework standardizes how Data Scientists build, train, evaluate, and deploy
ML models while abstracting infrastructure concerns.

Step-based API (recommended):
    from mlops_framework import PreprocessStep, TrainStep, InferenceStep, LocalRunner

Direct debugging (run from step file):
    if __name__ == "__main__":
        from mlops_framework import run_local
        run_local(ChurnPreprocess)

Legacy API (deprecated):
    from mlops_framework import TrainingPipeline, InferencePipeline
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Type, Union

import yaml

__version__ = "0.1.0"

# Step-based API (recommended)
from mlops_framework.core import (
    BaseStep,
    ExecutionContext,
    PreprocessStep,
    TrainStep,
    InferenceStep,
)
from mlops_framework.backends.execution import LocalRunner

# Legacy API (deprecated - use step-based model instead)
from mlops_framework.core import (
    BaseModel,
    TrainingPipeline,
    InferencePipeline,
    Runtime,
)
from mlops_framework.artifacts import ArtifactManager, ArtifactType
from mlops_framework.config import ConfigLoader


def _merge_step_config(full_config: dict, step_id: str, env: str) -> dict:
    """Merge steps[step_id] with environments[env][step_id] for step-specific config."""
    steps_cfg = full_config.get("steps", {})
    step_base = steps_cfg.get(step_id, {}).copy()
    env_overrides = full_config.get("environments", {}).get(env, {}).get(step_id, {})
    for k, v in env_overrides.items():
        if isinstance(v, dict) and k in step_base and isinstance(step_base[k], dict):
            step_base[k] = {**step_base[k], **v}
        else:
            step_base[k] = v
    return step_base


def run_local(
    step_class_or_path: Union[Type[BaseStep], str],
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    step_id: Optional[str] = None,
    env: str = "dev",
    tracking: bool = False,
    tracking_backend: str = "noop",
) -> None:
    """
    Run a step locally with default backends. For debugging and development.

    Use when running a step file directly:
        python -m steps.preprocess

    Args:
        step_class_or_path: Dotted path (e.g. "steps.preprocess.PartFailurePreprocess")
        config_path: Optional path to config YAML (default: pipeline/config.yaml)
        project_root: Optional project root (default: cwd)
        step_id: Optional step ID for per-step config (inferred from path if steps.X)
        env: Environment for config merge (dev, qa, prod)
        tracking: If True, persist metrics/params (default False)
        tracking_backend: noop, local, or vertex
    """
    project_root = Path(project_root or os.getcwd()).resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Resolve step class (always use string path so imports resolve)
    if isinstance(step_class_or_path, str):
        path = step_class_or_path
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        step_class = getattr(module, class_name)
        if step_id is None and path.startswith("steps."):
            step_id = path.split(".")[1]
    else:
        step_class = step_class_or_path

    # Load config: pipeline/config.yaml, merge per-step + env
    cfg_path = Path(config_path or project_root / "pipeline" / "config.yaml")
    if not cfg_path.exists():
        cfg_path = project_root / "config.yaml"
    full_config = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            full_config = yaml.safe_load(f) or {}
    config = _merge_step_config(full_config, step_id, env) if step_id else {}

    tb = full_config.get("tracking_backend", tracking_backend)
    if tracking and tb == "noop":
        tb = "local"
    runner = LocalRunner(
        artifacts_path=str(project_root / "artifacts"),
        runs_path=str(project_root / "runs"),
        tracking=tracking,
        tracking_backend=tb,
    )
    runner.run_step(step_class, config=config)


__all__ = [
    "BaseStep",
    "ExecutionContext",
    "PreprocessStep",
    "TrainStep",
    "InferenceStep",
    "LocalRunner",
    "run_local",
    "BaseModel",
    "TrainingPipeline",
    "InferencePipeline",
    "Runtime",
    "ArtifactManager",
    "ArtifactType",
    "ConfigLoader",
]
