"""Workflow orchestrator with profile-driven infrastructure resolution."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mlplatform.config.loader import load_workflow_config
from mlplatform.config.schema import ModelConfig, WorkflowConfig
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.invocation.base import InvocationStrategy
from mlplatform.log import get_logger
from mlplatform.profiles.registry import Profile, get_profile


def dev_context(
    dag_path: str | Path,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
) -> ExecutionContext:
    """Build an ExecutionContext for local development and debugging.

    Call this from a trainer/predictor's ``if __name__ == "__main__"`` block
    so you can run/debug the file directly::

        if __name__ == "__main__":
            from mlplatform.runner import dev_context
            ctx = dev_context("template_training_dag.yaml")
            trainer = MyTrainer()
            trainer.context = ctx
            trainer.train()
    """
    workflow = load_workflow_config(dag_path)
    model_cfg = workflow.models[model_index]
    return _build_context(workflow, model_cfg, profile, version, base_path)


def run_workflow(
    dag_path: str | Path,
    profile: str = "local",
    version: str | None = None,
    base_path: str | None = None,
) -> dict[str, str]:
    """Run all models defined in a DAG workflow config.

    Returns a dict mapping model_name -> result status.
    """
    workflow = load_workflow_config(dag_path)
    version = version or _generate_version()
    prof = get_profile(profile)
    log = get_logger("mlplatform.runner", workflow.log_level)
    log.info("Running workflow '%s' (%s) profile=%s version=%s",
             workflow.workflow_name, workflow.pipeline_type, profile, version)

    results: dict[str, str] = {}
    for model_cfg in workflow.models:
        ctx = _build_context(workflow, model_cfg, profile, version, base_path)
        try:
            if workflow.pipeline_type == "training":
                _run_training(model_cfg, ctx)
            else:
                invocation = prof.invocation_strategy_factory()
                _run_prediction(model_cfg, ctx, invocation)
            results[model_cfg.model_name] = "ok"
        except Exception as exc:
            ctx.log.error("Model '%s' failed: %s", model_cfg.model_name, exc)
            results[model_cfg.model_name] = f"error: {exc}"
    return results


def _generate_version() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"{ts}_{short_id}"


def _build_context(
    workflow: WorkflowConfig,
    model_cfg: ModelConfig,
    profile: str,
    version: str,
    base_path: str | None,
) -> ExecutionContext:
    prof = get_profile(profile)
    base = base_path or "./artifacts"
    storage = prof.storage_factory(base)
    tracker = prof.tracker_factory(base)
    log = get_logger(f"mlplatform.{model_cfg.model_name}", workflow.log_level)
    return ExecutionContext(
        storage=storage,
        experiment_tracker=tracker,
        feature_name=workflow.feature_name,
        model_name=model_cfg.model_name,
        version=version,
        optional_configs=model_cfg.optional_configs,
        log=log,
        _pipeline_type=workflow.pipeline_type,
    )


def _resolve_class(module_path: str, base_class: type) -> type:
    """Import a module and find the first subclass of base_class."""
    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
            return attr
    raise ImportError(f"No {base_class.__name__} subclass found in {module_path}")


def _run_training(model_cfg: ModelConfig, ctx: ExecutionContext) -> None:
    trainer_cls = _resolve_class(model_cfg.module, BaseTrainer)
    trainer = trainer_cls()
    trainer.context = ctx
    ctx.log.info("Starting training: %s", model_cfg.model_name)
    trainer.train()
    ctx.log.info("Training complete: %s", model_cfg.model_name)


def _run_prediction(
    model_cfg: ModelConfig,
    ctx: ExecutionContext,
    invocation: InvocationStrategy,
) -> Any:
    predictor_cls = _resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx
    return invocation.invoke(predictor, ctx)
