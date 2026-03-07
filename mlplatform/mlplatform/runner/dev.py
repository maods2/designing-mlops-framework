"""Development helpers — convenience functions for local development/debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlplatform.config.loader import load_workflow_config
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.profiles.registry import get_profile
from mlplatform.runner.resolve import resolve_class
from mlplatform.runner.workflow import _build_context


def dev_context(
    dag_path: str | Path,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    commit_hash: str | None = None,
    config_names: list[str] | None = None,
) -> ExecutionContext:
    """Build an ExecutionContext for local development and debugging.

    Call this from a trainer/predictor's ``if __name__ == "__main__"`` block
    so you can run/debug the file directly::

        if __name__ == "__main__":
            from mlplatform.runner import dev_context
            ctx = dev_context("example_model/pipeline/train.yaml")
            trainer = MyTrainer()
            trainer.context = ctx
            trainer.train()
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    model_cfg = workflow.models[model_index]
    return _build_context(workflow, model_cfg, profile, version, base_path, commit_hash)


def dev_predict(
    dag_path: str | Path,
    data: Any = None,
    model_index: int = 0,
    profile: str = "local",
    version: str = "dev",
    base_path: str | None = None,
    config_names: list[str] | None = None,
) -> Any:
    """Run prediction locally for development and debugging.

    When *data* is provided (a DataFrame), the framework I/O is skipped and
    ``predict`` is called directly.  When *data* is ``None``, the
    profile's ``InvocationStrategy`` handles data loading from the
    YAML-configured source.

    Switch between in-process and PySpark by passing *profile*:
    ``"local"`` (in-process) or ``"local-spark"`` (PySpark mapInPandas).
    """
    workflow = load_workflow_config(dag_path, config_names=config_names)
    model_cfg = workflow.models[model_index]
    ctx = _build_context(workflow, model_cfg, profile, version, base_path)

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
    invocation = prof.invocation_strategy_factory()
    return invocation.invoke(predictor, ctx, model_cfg)
