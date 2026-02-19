"""Local pipeline execution - load config and run steps without Airflow."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from mlplatform.config.loader import (
    _env_data_to_config,
    load_pipeline_config as _load_pipeline_config,
)
from mlplatform.config.registry import get_etb, get_runner, get_storage
from mlplatform.config.schema import PipelineConfig, RunConfig, StepConfig
from mlplatform.core.context import ExecutionContext
from mlplatform.core.steps import Step


def load_pipeline_config(
    dag_path: str | Path,
    steps_dir: str | Path,
    env: str,
    version: str | None = None,
) -> PipelineConfig:
    """Load and merge DAG and step configuration. Env configs defined in step YAMLs."""
    return _load_pipeline_config(dag_path, steps_dir, env, None, version)


def _instantiate_step(step_config: StepConfig) -> Step:
    """Load step class from module and instantiate."""
    mod = importlib.import_module(step_config.module)
    cls = getattr(mod, step_config.class_name)
    return cls()


def _build_context(
    run_config: RunConfig,
    cwd: Path | None = None,
    project_root: str | Path | None = None,
    base_path: str | Path | None = None,
) -> ExecutionContext:
    """Build ExecutionContext from RunConfig.
    base_path: Injected by orchestrator (bucket or root folder). Required for storage/etb.
    project_root: Model project root (e.g. template_model/). Used for packaging, etc.
    """
    env = run_config.env_config
    if project_root is not None:
        env.extra = {**env.extra, "project_root": str(Path(project_root).resolve())}
    # base_path: orchestrator-injected. Fallback for local: project_root/artifacts
    resolved_base = base_path
    if resolved_base is None and project_root is not None:
        resolved_base = str(Path(project_root).resolve() / "artifacts")
    if resolved_base is None:
        resolved_base = "./artifacts"
    resolved_base = str(Path(resolved_base).resolve()) if resolved_base else "./artifacts"
    runner_kwargs = env.extra.get("runner_config", {})
    storage = get_storage(env.storage, base_path=resolved_base)
    etb = get_etb(env.etb, base_path=resolved_base)
    runner = get_runner(env.runner, **runner_kwargs)

    return ExecutionContext(
        storage=storage,
        etb=etb,
        runner=runner,
        run_config=run_config,
        feature=run_config.feature,
        model_name=run_config.model_name,
        version=run_config.version,
        step_name=run_config.step.name,
        custom=run_config.custom,
    )


def run_step_local(
    step_name: str,
    dag_path: str | Path,
    steps_dir: str | Path,
    env: str = "dev",
    version: str | None = None,
    project_root: str | Path | None = None,
    base_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Run a single step locally. Kwargs are passed to step.run()."""
    config = load_pipeline_config(dag_path, steps_dir, env, version)
    step_config = next((s for s in config.steps if s.name == step_name), None)
    if step_config is None:
        raise ValueError(f"Step '{step_name}' not found in pipeline")

    env_data = step_config.envs.get(config.env) or step_config.envs.get("dev") or {}
    env_config = _env_data_to_config(env_data)

    run_config = RunConfig(
        step=step_config,
        pipeline_name=config.pipeline_name,
        model_name=config.model_name,
        version=config.version,
        feature=config.feature,
        env_config=env_config,
        custom=step_config.custom,
    )

    context = _build_context(run_config, project_root=project_root, base_path=base_path)
    step = _instantiate_step(step_config)
    step._context = context
    return context.runner.run(step, context, **kwargs)


def run_pipeline_local(
    config: PipelineConfig,
    project_root: str | Path | None = None,
    base_path: str | Path | None = None,
    **step_kwargs: Any,
) -> dict[str, Any]:
    """Run full pipeline locally. step_kwargs keyed by step name passed to each step."""
    results: dict[str, Any] = {}
    for step_config in config.steps:
        env_data = step_config.envs.get(config.env) or step_config.envs.get("dev") or {}
        env_config = _env_data_to_config(env_data)
        run_config = RunConfig(
            step=step_config,
            pipeline_name=config.pipeline_name,
            model_name=config.model_name,
            version=config.version,
            feature=config.feature,
            env_config=env_config,
            custom=step_config.custom,
        )
        context = _build_context(run_config, project_root=project_root, base_path=base_path)
        step = _instantiate_step(step_config)
        step._context = context
        kwargs = step_kwargs.get(step_config.name, {})
        result = context.runner.run(step, context, **kwargs)
        results[step_config.name] = result
    return results
