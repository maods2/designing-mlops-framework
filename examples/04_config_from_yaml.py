"""Example: mlplatform.config — load a PipelineConfig from a YAML file.

Demonstrates PipelineConfig.from_yaml() with both a training and a prediction
pipeline.  Pipeline YAML files are in examples/pipelines/.

Install
-------
    pip install mlplatform[config]
    # or, from this repo:
    pip install -e "mlplatform/[config]"

Run
---
    python examples/04_config_from_yaml.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — only needed when running directly from the repo without
# a pip install.  Safe to remove if the package is installed.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
_mlplatform_src = _repo_root / "mlplatform"
for _p in [str(_repo_root), str(_mlplatform_src)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from mlplatform.config import PipelineConfig, PredictionConfig, TrainingConfig  # noqa: E402

PIPELINES_DIR = Path(__file__).resolve().parent / "pipelines"


# ── 1. Load a training pipeline ───────────────────────────────────────────────

print("=" * 60)
print("1. Load a training pipeline from YAML")
print("=" * 60)

train_pipeline = PipelineConfig.from_yaml(PIPELINES_DIR / "train.yaml")

print(f"\nworkflow_name  : {train_pipeline.workflow_name}")
print(f"pipeline_type  : {train_pipeline.pipeline_type}")
print(f"feature_name   : {train_pipeline.feature_name}")
print(f"execution_mode : {train_pipeline.execution_mode}")
print(f"config_version : {train_pipeline.config_version}")

print("\n--- Computed fields ---")
print(f"is_training    : {train_pipeline.is_training}")   # True
print(f"is_prediction  : {train_pipeline.is_prediction}") # False
print(f"model_count    : {train_pipeline.model_count}")   # 2

print("\n--- Models ---")
for i, model in enumerate(train_pipeline.models):
    assert isinstance(model, TrainingConfig)
    print(f"\n  Model [{i}]: {model.model_name}")
    print(f"    module            : {model.module}")
    print(f"    platform          : {model.platform}")
    print(f"    model_version     : {model.model_version}")
    print(f"    optional_configs  : {model.optional_configs}")
    print(f"    artifact_base_path: {model.artifact_base_path!r}")
    print(f"    is_cloud_training : {model.is_cloud_training}")


# ── 2. Load a prediction pipeline ────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Load a prediction pipeline from YAML")
print("=" * 60)

pred_pipeline = PipelineConfig.from_yaml(PIPELINES_DIR / "predict.yaml")

print(f"\nworkflow_name  : {pred_pipeline.workflow_name}")
print(f"pipeline_type  : {pred_pipeline.pipeline_type}")

print("\n--- Computed fields ---")
print(f"is_training    : {pred_pipeline.is_training}")    # False
print(f"is_prediction  : {pred_pipeline.is_prediction}")  # True
print(f"model_count    : {pred_pipeline.model_count}")    # 1

print("\n--- Models ---")
for i, model in enumerate(pred_pipeline.models):
    assert isinstance(model, PredictionConfig)
    print(f"\n  Model [{i}]: {model.model_name}")
    print(f"    module              : {model.module}")
    print(f"    model_version       : {model.model_version}")
    print(f"    input_source        : {model.input_source!r}")
    print(f"    bigquery_table_ref  : {model.bigquery_table_ref!r}")
    print(f"    has_output_table    : {model.has_output_table}")
    print(f"    is_cloud_prediction : {model.is_cloud_prediction}")
    print(f"    artifact_base_path  : {model.artifact_base_path!r}")
    print(f"    optional_configs    : {model.optional_configs}")


# ── 3. Override config profiles at load time ─────────────────────────────────

print("\n" + "=" * 60)
print("3. Override config profiles at load time")
print("=" * 60)

# config_names overrides whatever 'config:' key is in the YAML.
# Useful for switching between environments without editing the file.
train_dev = PipelineConfig.from_yaml(
    PIPELINES_DIR / "train.yaml",
    config_names=["global", "dev"],
)
print(f"\nLoaded with config_names=['global', 'dev']")
print(f"config_profiles recorded: {train_dev.config_profiles}")


# ── 4. Iterate models generically ────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Iterate models generically — check type at runtime")
print("=" * 60)

for pipeline in [train_pipeline, pred_pipeline]:
    print(f"\nPipeline: {pipeline.workflow_name!r}")
    for model in pipeline.models:
        if isinstance(model, TrainingConfig):
            print(f"  [training]   {model.model_name} → {model.artifact_base_path}")
        elif isinstance(model, PredictionConfig):
            print(f"  [prediction] {model.model_name} source={model.input_source!r}")
