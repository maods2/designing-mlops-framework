"""Workflow orchestrator with profile-driven infrastructure resolution."""

from __future__ import annotations

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
from mlplatform.profiles.registry import get_profile
from mlplatform.runner.resolve import resolve_class
from mlplatform.utils.logging import get_logger


def run_workflow(
    dag_path: str | Path,
    profile: str = "local",
    version: str | None = None,
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
) -> dict[str, str]:
    """Run all models defined in a DAG workflow config.

    Args:
        dag_path: Path to the DAG YAML file.
        profile: Infrastructure profile name.
        version: Model version string (auto-generated if omitted).
        base_path: Artifact storage base path.
        commit_hash: Git commit hash for reproducibility tracking.
        config_names: Config profile names to merge (overrides DAG ``config:`` key).

    Returns:
        Dict mapping model_name -> result status ("ok" or "error: <msg>").
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    version = version or _generate_version()
    prof = get_profile(profile)
    log = get_logger("mlplatform.runner", workflow.log_level)
    log.info("Running workflow '%s' (%s) profile=%s version=%s",
             workflow.workflow_name, workflow.pipeline_type, profile, version)
    if workflow.config_profiles:
        log.info("Config profiles loaded: %s", workflow.config_profiles)

    results: dict[str, str] = {}
    for model_cfg in workflow.models:
        ctx = _build_context(workflow, model_cfg, profile, version, base_path, commit_hash)
        _log_framework_params(ctx, profile)
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
    commit_hash: str | None = None,
) -> ExecutionContext:
    prof = get_profile(profile)
    return ExecutionContext.from_profile(
        profile=prof,
        feature_name=workflow.feature_name,
        model_name=model_cfg.model_name,
        version=version,
        base_path=base_path or "./artifacts",
        pipeline_type=workflow.pipeline_type,
        log_level=workflow.log_level,
        optional_configs=model_cfg.optional_configs,
        commit_hash=commit_hash,
    )


def _log_framework_params(ctx: ExecutionContext, profile: str) -> None:
    """Log framework-level parameters for reproducibility tracking."""
    params: dict[str, Any] = {
        "mlplatform.profile": profile,
        "mlplatform.version": ctx.version,
        "mlplatform.pipeline_type": ctx._pipeline_type,
    }
    if ctx.commit_hash:
        params["mlplatform.commit_hash"] = ctx.commit_hash
    ctx.log_params(params)


def _run_training(model_cfg: ModelConfig, ctx: ExecutionContext) -> None:
    trainer_cls = resolve_class(model_cfg.module, BaseTrainer)
    trainer = trainer_cls()
    trainer.context = ctx
    trainer.setup()
    ctx.log.info("Starting training: %s", model_cfg.model_name)
    try:
        trainer.train()
        ctx.log.info("Training complete: %s", model_cfg.model_name)
    finally:
        trainer.teardown()


def _run_prediction(
    model_cfg: ModelConfig,
    ctx: ExecutionContext,
    invocation: InvocationStrategy,
) -> Any:
    predictor_cls = resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx
    predictor.setup()
    try:
        return invocation.invoke(predictor, ctx, model_cfg)
    finally:
        predictor.teardown()
