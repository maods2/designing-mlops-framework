"""Configuration loading from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mlplatform.config.schema import (
    EnvConfig,
    ModelConfig,
    PipelineConfig,
    StepConfig,
    WorkflowConfig,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# New format: WorkflowConfig (from template_training_dag / template_prediction_dag)
# ---------------------------------------------------------------------------


def load_workflow_config(dag_path: str | Path) -> WorkflowConfig:
    """Load a workflow from the new DAG template format."""
    dag_path = Path(dag_path)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    data = _load_yaml(dag_path)
    pipeline_type = data.get("pipeline_type", "training")

    models: list[ModelConfig] = []
    for entry in data.get("models", []):
        platform = entry.get("training_platform") or entry.get("serving_platform") or "VertexAI"
        models.append(ModelConfig(
            model_name=entry["model_name"],
            module=entry.get("module", ""),
            compute=entry.get("compute", "s"),
            platform=platform,
            optional_configs=entry.get("optional_configs") or {},
            prediction_dataset_name=entry.get("prediction_dataset_name"),
            prediction_table_name=entry.get("prediction_table_name"),
            model_id=entry.get("model_id"),
            model_version=entry.get("model_version", "latest"),
            prediction_output_dataset_table=entry.get("prediction_output_dataset_table"),
            predicted_label_column_name=entry.get("predicted_label_column_name"),
            predicted_timestamp_column_name=entry.get("predicted_timestamp_column_name"),
            predicted_probability_column_name=entry.get("predicted_probability_column_name"),
            unique_identifier_column=entry.get("unique_identifier_column"),
        ))

    return WorkflowConfig(
        workflow_name=data.get("workflow_name", "default_workflow"),
        execution_mode=data.get("execution_mode", "sequential"),
        pipeline_type=pipeline_type,
        feature_name=data.get("feature_name", "default"),
        config_version=data.get("config_version", 2),
        models=models,
    )


# ---------------------------------------------------------------------------
# Legacy format: PipelineConfig (old DAG + step YAML files)
# ---------------------------------------------------------------------------


def _env_data_to_config(env_data: dict[str, Any]) -> EnvConfig:
    extra = {k: v for k, v in env_data.items() if k not in ("runner", "storage", "etb", "serving_mode", "base_path")}
    return EnvConfig(
        runner=env_data.get("runner", "LocalRunner"),
        storage=env_data.get("storage", "LocalFileSystem"),
        etb=env_data.get("etb", "LocalJsonTracker"),
        serving_mode=env_data.get("serving_mode", "ProceduralLocal"),
        base_path=None,
        extra=extra,
    )


def load_step_config(step_name: str, steps_dir: str | Path) -> StepConfig:
    """Legacy: Load step configuration from steps directory."""
    path = Path(steps_dir) / f"{step_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Step config not found: {path}")
    data = _load_yaml(path)
    core_keys = {"name", "type", "module", "class", "envs"}
    custom = {k: v for k, v in data.items() if k not in core_keys}
    envs = data.get("envs") or {}
    return StepConfig(
        name=data["name"],
        type=data["type"],
        module=data["module"],
        class_name=data["class"],
        envs=envs,
        custom=custom,
    )


def load_pipeline_config(
    dag_path: str | Path,
    steps_dir: str | Path,
    env: str,
    envs_dir: str | Path | None = None,
    version: str | None = None,
) -> PipelineConfig:
    """Legacy: Load and merge DAG, step, and environment configuration."""
    dag_path = Path(dag_path)
    steps_dir = Path(steps_dir)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    dag_data = _load_yaml(dag_path)
    pipeline_data = dag_data.get("pipeline", dag_data)
    pipeline_name = pipeline_data.get("name", "default_pipeline")
    model_name = pipeline_data.get("model_name", "default_model")
    version = version or pipeline_data.get("version")
    if not version or str(version).lower() in ("null", "none", ""):
        import uuid
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        version = f"{ts}_{short_id}"
    feature = pipeline_data.get("feature") or pipeline_name

    steps: list[StepConfig] = []
    for step_entry in pipeline_data.get("steps", []):
        if isinstance(step_entry, str):
            step_name = step_entry
            dag_overrides = {}
        else:
            step_name = step_entry.get("name", "unknown")
            dag_overrides = {k: v for k, v in step_entry.items() if k != "custom"}
            dag_overrides.pop("name", None)
        step_config = load_step_config(step_name, steps_dir)
        if dag_overrides:
            for k, v in dag_overrides.items():
                if k in ("type", "module", "class"):
                    setattr(step_config, "class_name" if k == "class" else k, v)
                else:
                    step_config.custom[k] = v
        if isinstance(step_entry, dict) and "custom" in step_entry:
            step_config.custom.update(step_entry["custom"])
        steps.append(step_config)

    return PipelineConfig(
        pipeline_name=pipeline_name,
        model_name=model_name,
        version=version,
        feature=feature,
        steps=steps,
        env=env,
    )
