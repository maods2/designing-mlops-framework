"""Flat workflow orchestrator replacing the local/, profiles/, runners/ hierarchy."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mlplatform.artifacts.local import LocalArtifactStore
from mlplatform.config.loader import load_workflow_config
from mlplatform.config.schema import ModelConfig, WorkflowConfig
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.log import get_logger
from mlplatform.storage.local import LocalFileSystem
from mlplatform.tracking.local import LocalJsonTracker


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
    log = get_logger("mlplatform.runner", workflow.log_level)
    log.info("Running workflow '%s' (%s) version=%s", workflow.workflow_name, workflow.pipeline_type, version)

    results: dict[str, str] = {}
    for model_cfg in workflow.models:
        ctx = _build_context(workflow, model_cfg, profile, version, base_path)
        try:
            if workflow.pipeline_type == "training":
                _run_training(model_cfg, ctx)
            else:
                _run_prediction(model_cfg, ctx)
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
    storage, artifact_store, tracker = _create_infra(profile, base_path)
    log = get_logger(f"mlplatform.{model_cfg.model_name}", workflow.log_level)
    return ExecutionContext(
        storage=storage,
        artifact_store=artifact_store,
        experiment_tracker=tracker,
        feature_name=workflow.feature_name,
        model_name=model_cfg.model_name,
        version=version,
        optional_configs=model_cfg.optional_configs,
        log=log,
        _pipeline_type=workflow.pipeline_type,
    )


def _create_infra(profile: str, base_path: str | None):
    base = base_path or "./artifacts"
    if profile in ("local", "local-spark", "cloud-batch-emulated"):
        return (LocalFileSystem(base), LocalArtifactStore(base), LocalJsonTracker(base))
    return (LocalFileSystem(base), LocalArtifactStore(base), LocalJsonTracker(base))


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


def _run_prediction(model_cfg: ModelConfig, ctx: ExecutionContext) -> None:
    predictor_cls = _resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx
    ctx.log.info("Loading model for prediction: %s", model_cfg.model_name)
    predictor.load_model()
    ctx.log.info("Model loaded: %s", model_cfg.model_name)
