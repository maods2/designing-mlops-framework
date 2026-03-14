"""Simplified runner — execute a single model from a frozen PipelineConfig."""

from __future__ import annotations

from typing import Any

from mlplatform.config.models import PipelineConfig
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.profiles.registry import get_profile
from mlplatform.runner.resolve import resolve_class


def execute(config: PipelineConfig) -> dict[str, str]:
    """Execute a single model training or prediction from a frozen config.

    This is the primary entry point for V3 — replaces ``run_workflow``.
    One model per invocation; orchestrators handle DAGs externally.
    """
    profile = get_profile(config.profile)
    ctx = ExecutionContext.from_config(config, profile)
    _log_framework_params(ctx, config)

    try:
        if config.pipeline_type == "training":
            _run_training(config, ctx)
        else:
            inference = profile.inference_strategy_factory()
            _run_prediction(config, ctx, inference)
        return {"status": "ok", "model_name": config.model_name}
    except Exception as exc:
        ctx.log.error("Model '%s' failed: %s", config.model_name, exc)
        return {"status": f"error: {exc}", "model_name": config.model_name}


def _log_framework_params(ctx: ExecutionContext, config: PipelineConfig) -> None:
    """Log framework-level parameters for reproducibility."""
    params: dict[str, Any] = {
        "mlplatform.profile": config.profile,
        "mlplatform.version": ctx.version,
        "mlplatform.pipeline_type": config.pipeline_type,
    }
    if ctx.commit_hash:
        params["mlplatform.commit_hash"] = ctx.commit_hash
    ctx.log_params(params)


def _run_training(config: PipelineConfig, ctx: ExecutionContext) -> None:
    if config.module:
        trainer_cls = resolve_class(config.module, BaseTrainer)
    else:
        raise ValueError("PipelineConfig.module is required for training execution")

    trainer = trainer_cls()
    trainer.context = ctx
    trainer.setup()
    ctx.log.info("Starting training: %s", config.model_name)
    try:
        trainer.train()
        ctx.log.info("Training complete: %s", config.model_name)
    finally:
        trainer.teardown()


def _run_prediction(
    config: PipelineConfig,
    ctx: ExecutionContext,
    inference: Any,
) -> Any:
    if config.module:
        predictor_cls = resolve_class(config.module, BasePredictor)
    else:
        raise ValueError("PipelineConfig.module is required for prediction execution")

    predictor = predictor_cls()
    predictor.context = ctx
    predictor.setup()
    try:
        return inference.invoke(predictor, ctx, config)
    finally:
        predictor.teardown()
