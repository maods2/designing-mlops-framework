"""Local workflow execution - load config and run models via profiles and resolver."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mlplatform.config.loader import load_workflow_config
from mlplatform.config.schema import ModelConfig, WorkflowConfig
from mlplatform.core.enums import ExecutionNature
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.steps import InferenceStep, TrainStep
from mlplatform.core.trainer import BaseTrainer
from mlplatform.profiles.registry import get_profile
from mlplatform.profiles.resolver import PrimitiveResolver


def _generate_version() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"{ts}_{short_id}"


def _resolve_class(module_path: str, base_class: type) -> type:
    """Import a module and find the first subclass of base_class."""
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, base_class)
            and attr is not base_class
        ):
            return attr
    raise ImportError(
        f"No {base_class.__name__} subclass found in module '{module_path}'"
    )


def run_workflow(
    dag_path: str | Path,
    profile_name: str = "local",
    version: str | None = None,
    project_root: str | Path | None = None,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a full workflow from a DAG YAML using the new architecture.

    Loads the WorkflowConfig, resolves profile, and runs each model
    through the PrimitiveResolver -> runner.execute() pipeline.
    """
    workflow = load_workflow_config(dag_path)
    version = version or _generate_version()

    resolved_base = str(Path(base_path).resolve()) if base_path else "./artifacts"
    if project_root is not None:
        resolved_base = resolved_base or str(Path(project_root).resolve() / "artifacts")

    profile = get_profile(profile_name, base_path=resolved_base)
    resolver = PrimitiveResolver()

    results: dict[str, Any] = {}
    for model_cfg in workflow.models:
        result = _run_model(
            workflow=workflow,
            model_cfg=model_cfg,
            profile=profile,
            resolver=resolver,
            version=version,
            project_root=project_root,
        )
        results[model_cfg.model_name] = result

    return results


def _run_model(
    workflow: WorkflowConfig,
    model_cfg: ModelConfig,
    profile: Any,
    resolver: PrimitiveResolver,
    version: str,
    project_root: str | Path | None = None,
) -> Any:
    """Run a single model entry from the workflow."""
    runtime_config = {
        "workflow_name": workflow.workflow_name,
        "pipeline_type": workflow.pipeline_type,
        "feature_name": workflow.feature_name,
        "model_name": model_cfg.model_name,
        "module": model_cfg.module,
        "version": version,
        "compute": model_cfg.compute,
        "platform": model_cfg.platform,
        "optional_configs": model_cfg.optional_configs,
        "model_version": model_cfg.model_version,
    }
    environment_metadata = {
        "base_path": str(getattr(profile.storage, "base_path", "./artifacts")),
        "execution_mode": workflow.execution_mode,
        "config_version": workflow.config_version,
    }
    if project_root is not None:
        environment_metadata["project_root"] = str(Path(project_root).resolve())

    if workflow.pipeline_type == "training":
        return _run_training_model(
            model_cfg, profile, resolver, runtime_config, environment_metadata
        )
    else:
        return _run_prediction_model(
            model_cfg, profile, resolver, runtime_config, environment_metadata
        )


def _run_training_model(
    model_cfg: ModelConfig,
    profile: Any,
    resolver: PrimitiveResolver,
    runtime_config: dict[str, Any],
    environment_metadata: dict[str, Any],
) -> Any:
    trainer_cls = _resolve_class(model_cfg.module, BaseTrainer)
    trainer = trainer_cls()

    step = TrainStep()
    runner, context = resolver.resolve(
        step=step,
        profile=profile,
        trainer=trainer,
        runtime_config=runtime_config,
        environment_metadata=environment_metadata,
    )

    trainer.context = context
    context.trainer = trainer
    return runner.execute(step, context)


def _run_prediction_model(
    model_cfg: ModelConfig,
    profile: Any,
    resolver: PrimitiveResolver,
    runtime_config: dict[str, Any],
    environment_metadata: dict[str, Any],
) -> Any:
    predictor_cls = _resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()

    step = InferenceStep(execution_nature=ExecutionNature.JOB)
    runner, context = resolver.resolve(
        step=step,
        profile=profile,
        predictor=predictor,
        runtime_config=runtime_config,
        environment_metadata=environment_metadata,
    )

    predictor.context = context
    context.predictor = predictor
    predictor.load_model()
    return runner.execute(step, context)
