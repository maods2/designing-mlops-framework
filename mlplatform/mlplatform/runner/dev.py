"""Development helpers — convenience functions for local development/debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlplatform.config.loader import load_workflow_config
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.profiles.registry import get_profile
from mlplatform.runner.resolve import resolve_class
from mlplatform.runner.workflow import _build_context, _log_framework_params


def dev_train(
    dag_path: str | Path,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> BaseTrainer:
    """Run training locally for development.

    Builds context, resolves trainer from DAG config, runs setup/train/teardown.
    Returns the trainer for inspection. For one-liner usage::

        if __name__ == "__main__":
            from mlplatform.runner import dev_train
            dev_train("example_model/pipeline/train.yaml")
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    model_cfg = workflow.models[model_index]
    ctx = _build_context(
        workflow, model_cfg, profile, version, base_path, commit_hash, extra_overrides
    )
    _log_framework_params(ctx, profile)

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
    return trainer


def dev_context(
    dag_path: str | Path,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> ExecutionContext:
    """Build an ExecutionContext for local development and debugging.

    For one-liner training, use :func:`dev_train` instead. Use ``dev_context`` when
    you need the context before creating the trainer (e.g. custom setup, tests)::

        if __name__ == "__main__":
            from mlplatform.runner import dev_context
            ctx = dev_context("example_model/pipeline/train.yaml")
            trainer = MyTrainer()
            trainer.context = ctx
            trainer.train()
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    model_cfg = workflow.models[model_index]
    return _build_context(
        workflow, model_cfg, profile, version, base_path, commit_hash, extra_overrides
    )


def dev_predict(
    dag_path: str | Path,
    data: Any = None,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    config_names: list[str] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> Any:
    """Run prediction locally for development and debugging.

    When *data* is provided (a DataFrame), the framework I/O is skipped and
    ``predict`` is called directly.  When *data* is ``None``, the
    profile's ``InferenceStrategy`` handles data loading from the
    YAML-configured source.

    Switch between in-process and PySpark by passing *profile*:
    ``"local"`` (in-process) or ``"local-spark"`` (PySpark mapInPandas).
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    model_cfg = workflow.models[model_index]
    ctx = _build_context(
        workflow, model_cfg, profile, version, base_path, extra_overrides=extra_overrides
    )

    predictor_cls = resolve_class(model_cfg.module, BasePredictor)
    predictor = predictor_cls()
    predictor.context = ctx
    predictor.load_model()
    ctx.log.info("Model loaded: %s", model_cfg.model_name)

    if data is not None:
        result = predictor.predict(data)
        ctx.log.info("Dev prediction complete (manual data): %d rows", len(result))
        return result

    prof = get_profile(profile)
    inference = prof.inference_strategy_factory()
    return inference.invoke(predictor, ctx, model_cfg)
