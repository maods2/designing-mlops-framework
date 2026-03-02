"""Workflow orchestrator with profile-driven infrastructure resolution."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.config.schema import ModelConfig, TaskConfig, UnifiedPipelineConfig
from mlplatform.core.artifact_registry import ArtifactRegistry
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.invocation.base import InvocationStrategy
from mlplatform.log import get_logger
from mlplatform.profiles.registry import Profile, get_profile


def dev_train_context(
    pipeline_path: str | Path,
    task_id: str = "train_model",
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
) -> ExecutionContext:
    """Build an ExecutionContext for a specific training task (local dev/debug).

    Call this from a trainer's ``if __name__ == "__main__"`` block::

        if __name__ == "__main__":
            from mlplatform.runner import dev_train_context
            ctx = dev_train_context("example_model/pipeline/train.yaml", task_id="train_model")
            trainer = MyTrainer()
            trainer.context = ctx
            trainer.train()
    """
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        pipeline_path, task_id=task_id, config_names=config_names
    )
    task_cfg = pipeline.tasks[0]
    return _build_context(pipeline, task_cfg, profile, version, base_path, commit_hash)


def dev_predict_context(
    pipeline_path: str | Path,
    data: Any = None,
    task_id: str = "predict",
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    config_names: list[str] | None = None,
) -> Any:
    """Run prediction for a specific task locally (dev/debug).

    When *data* is provided (a DataFrame), the framework I/O is skipped and
    ``predict`` is called directly.  When *data* is ``None``, the
    profile's ``InvocationStrategy`` handles data loading from the
    YAML-configured source.

    Switch between in-process and PySpark by passing *profile*:
    ``"local"`` (in-process) or ``"local-spark"`` (PySpark mapInPandas).
    """
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        pipeline_path, task_id=task_id, config_names=config_names
    )
    task_cfg = pipeline.tasks[0]
    model_cfg = task_cfg.to_model_config()
    ctx = _build_context(pipeline, task_cfg, profile, version, base_path)

    predictor_cls = _resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx
    predictor.load_model()
    ctx.log.info("Model loaded: %s", model_cfg.model_name)

    if data is not None:
        result = predictor.predict(data)
        ctx.log.info("Dev prediction complete (manual data): %d rows", len(result))
        return result

    prof = get_profile(profile)
    invocation = prof.invocation_strategy_factory()
    return invocation.invoke(predictor, ctx, model_cfg)


# Backward-compatible aliases
dev_context = dev_train_context
dev_predict = dev_predict_context


def run_workflow(
    dag_path: str | Path,
    task_id: str | None = None,
    profile: str = "local",
    version: str | None = None,
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
) -> dict[str, str]:
    """Run pipeline tasks. When task_id is given, runs only that task.

    Args:
        dag_path: Path to the pipeline YAML file.
        task_id: If provided, run only this task. Otherwise run all executable tasks.
        profile: Infrastructure profile name.
        version: Model version string (auto-generated if omitted).
        base_path: Artifact storage base path.
        commit_hash: Git commit hash for reproducibility tracking.
        config_names: Config profile names to merge (overrides task ``config:`` key).

    Returns:
        Dict mapping model_name -> result status ("ok" or "error: <msg>").
    """
    pipeline = ConfigLoaderFactory.load_pipeline_config(
        dag_path, task_id=task_id, config_names=config_names
    )
    version = version or _generate_version()
    prof = get_profile(profile)
    log = get_logger("mlplatform.runner", pipeline.log_level)
    log.info(
        "Running pipeline '%s' (%s) profile=%s version=%s",
        pipeline.pipeline_name,
        pipeline.pipeline_type,
        profile,
        version,
    )
    if pipeline.config_profiles:
        log.info("Config profiles loaded: %s", pipeline.config_profiles)

    # Only run tasks that have a module (executable)
    executable_tasks = [t for t in pipeline.tasks if t.module]
    if not executable_tasks:
        log.warning("No executable tasks (with module) found in pipeline")
        return {}

    results: dict[str, str] = {}
    for task_cfg in executable_tasks:
        ctx = _build_context(pipeline, task_cfg, profile, version, base_path, commit_hash)
        model_cfg = task_cfg.to_model_config()
        _log_framework_params(ctx, profile)
        try:
            if pipeline.pipeline_type == "training":
                _run_training(model_cfg, ctx)
            else:
                invocation = prof.invocation_strategy_factory()
                _run_prediction(model_cfg, ctx, invocation)
            results[model_cfg.model_name] = "ok"
        except Exception as exc:
            ctx.log.error("Task '%s' failed: %s", task_cfg.task_id, exc)
            results[model_cfg.model_name] = f"error: {exc}"
    return results


def _generate_version() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"{ts}_{short_id}"


def _build_context(
    pipeline: UnifiedPipelineConfig,
    task_cfg: TaskConfig,
    profile: str,
    version: str,
    base_path: str | None,
    commit_hash: str | None = None,
) -> ExecutionContext:
    prof = get_profile(profile)
    base = base_path or "./artifacts"
    storage = prof.storage_factory(base)
    tracker = prof.tracker_factory(base)
    model_name = task_cfg.model_name or task_cfg.task_id
    log = get_logger(f"mlplatform.{model_name}", pipeline.log_level)
    registry = ArtifactRegistry(
        storage=storage,
        feature_name=pipeline.feature_name,
        model_name=model_name,
        version=version,
    )
    return ExecutionContext(
        artifacts=registry,
        experiment_tracker=tracker,
        feature_name=pipeline.feature_name,
        model_name=model_name,
        version=version,
        optional_configs=task_cfg.optional_configs,
        log=log,
        _pipeline_type=pipeline.pipeline_type,
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
    return invocation.invoke(predictor, ctx, model_cfg)
