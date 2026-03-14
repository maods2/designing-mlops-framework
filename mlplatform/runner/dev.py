"""Development helpers — convenience functions for local development/debugging."""

from __future__ import annotations

from typing import Any

from mlplatform.config.loader import load_model_config
from mlplatform.config.models import PipelineConfig
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.profiles.registry import get_profile
from mlplatform.runner.execute import _log_framework_params
from mlplatform.runner.resolve import resolve_class


def dev_train(
    config: PipelineConfig | None = None,
    *,
    model_name: str | None = None,
    feature: str | None = None,
    trainer_class: type[BaseTrainer] | None = None,
    version: str = "dev",
    profile: str = "local",
    config_list: list[str] | None = None,
    config_dir: str = "./config",
) -> BaseTrainer:
    """Run training locally for development.

    Accepts a PipelineConfig directly, or builds one from args + config template.
    Returns the trainer for inspection.

    Example::

        # With PipelineConfig
        config = PipelineConfig.from_dict({...})
        dev_train(config)

        # With args (auto-loads config template)
        dev_train(model_name="churn_model", feature="churn")
    """
    if config is None:
        merged = load_model_config(config_list, config_dir)
        config = PipelineConfig.from_dict({
            "model_name": model_name or merged.get("model_name", "default"),
            "feature": feature or merged.get("feature", "default"),
            "version": version,
            "profile": profile,
            "pipeline_type": "training",
            "user_config": merged,
        })

    prof = get_profile(config.profile)
    ctx = ExecutionContext.from_config(config, prof)
    _log_framework_params(ctx, config)

    if trainer_class is not None:
        trainer = trainer_class()
    elif config.module:
        trainer_cls = resolve_class(config.module, BaseTrainer)
        trainer = trainer_cls()
    else:
        raise ValueError(
            "Either pass trainer_class= or set module in PipelineConfig"
        )

    trainer.context = ctx
    trainer.setup()
    ctx.log.info("Starting training: %s", config.model_name)
    try:
        trainer.train()
        ctx.log.info("Training complete: %s", config.model_name)
    finally:
        trainer.teardown()
    return trainer


def dev_context(
    config: PipelineConfig | None = None,
    *,
    model_name: str | None = None,
    feature: str | None = None,
    version: str = "dev",
    profile: str = "local",
    config_list: list[str] | None = None,
    config_dir: str = "./config",
) -> ExecutionContext:
    """Build an ExecutionContext for local development and debugging.

    Example::

        ctx = dev_context(model_name="churn_model", feature="churn")
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()
    """
    if config is None:
        merged = load_model_config(config_list, config_dir)
        config = PipelineConfig.from_dict({
            "model_name": model_name or merged.get("model_name", "default"),
            "feature": feature or merged.get("feature", "default"),
            "version": version,
            "profile": profile,
            "pipeline_type": "training",
            "user_config": merged,
        })

    prof = get_profile(config.profile)
    return ExecutionContext.from_config(config, prof)


def dev_predict(
    config: PipelineConfig | None = None,
    data: Any = None,
    *,
    model_name: str | None = None,
    feature: str | None = None,
    predictor_class: type[BasePredictor] | None = None,
    version: str = "dev",
    profile: str = "local",
    config_list: list[str] | None = None,
    config_dir: str = "./config",
) -> Any:
    """Run prediction locally for development and debugging.

    When *data* is provided (a DataFrame), the framework I/O is skipped and
    predict() is called directly. When *data* is None, the profile's
    InferenceStrategy handles data loading.
    """
    if config is None:
        merged = load_model_config(config_list, config_dir)
        config = PipelineConfig.from_dict({
            "model_name": model_name or merged.get("model_name", "default"),
            "feature": feature or merged.get("feature", "default"),
            "version": version,
            "profile": profile,
            "pipeline_type": "prediction",
            "user_config": merged,
        })

    prof = get_profile(config.profile)
    ctx = ExecutionContext.from_config(config, prof)

    if predictor_class is not None:
        predictor = predictor_class()
    elif config.module:
        predictor_cls = resolve_class(config.module, BasePredictor)
        predictor = predictor_cls()
    else:
        raise ValueError(
            "Either pass predictor_class= or set module in PipelineConfig"
        )

    predictor.context = ctx
    predictor.setup()
    predictor.load_model()
    ctx.log.info("Model loaded: %s", config.model_name)

    if data is not None:
        result = predictor.predict(data)
        ctx.log.info("Dev prediction complete (manual data): %d rows", len(result))
        predictor.teardown()
        return result

    inference = prof.inference_strategy_factory()
    try:
        return inference.invoke(predictor, ctx, config)
    finally:
        predictor.teardown()
