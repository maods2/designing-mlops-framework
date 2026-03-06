"""Example: mlplatform.config — build TrainingConfig and PredictionConfig directly.

Demonstrates constructing config objects in Python (without a YAML file) and
inspecting their computed fields.  Useful for programmatic pipeline construction
and for understanding what each field does.

Install
-------
    pip install mlplatform[config]
    # or, from this repo:
    pip install -e "mlplatform/[config]"

Run
---
    python examples/03_config_direct.py
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

from mlplatform.config import PredictionConfig, TrainingConfig  # noqa: E402


# ── 1. TrainingConfig ────────────────────────────────────────────────────────

print("=" * 60)
print("1. TrainingConfig")
print("=" * 60)

train = TrainingConfig(
    model_name="churn_model",
    module="churn.train",
    feature_name="churn",
    model_version="1.0",
    platform="VertexAI",
    optional_configs={
        "learning_rate": 0.01,
        "max_iter": 500,
        "test_size": 0.2,
    },
)

print(f"\nmodel_name        : {train.model_name}")
print(f"module            : {train.module}")
print(f"platform          : {train.platform}")
print(f"model_version     : {train.model_version}")
print(f"optional_configs  : {train.optional_configs}")

print("\n--- Computed fields ---")
print(f"artifact_base_path: {train.artifact_base_path!r}")
# → 'churn/churn_model/1.0'
print(f"is_cloud_training : {train.is_cloud_training}")
# → True

# Local platform → is_cloud_training is False
local_train = TrainingConfig(
    model_name="churn_model",
    module="churn.train",
    platform="local",
    model_version="dev",
)
print(f"\nLocal platform → artifact_base_path: {local_train.artifact_base_path!r}")
# → None  (feature_name not set)
print(f"Local platform → is_cloud_training : {local_train.is_cloud_training}")
# → False

# With feature_name on a local run
local_train_with_feature = TrainingConfig(
    model_name="churn_model",
    module="churn.train",
    feature_name="churn",
    platform="local",
    model_version="dev",
)
print(f"\nLocal + feature   → artifact_base_path: {local_train_with_feature.artifact_base_path!r}")
# → 'churn/churn_model/dev'


# ── 2. PredictionConfig — BigQuery source ────────────────────────────────────

print("\n" + "=" * 60)
print("2. PredictionConfig — BigQuery source")
print("=" * 60)

pred_bq = PredictionConfig(
    model_name="churn_model",
    module="churn.predict",
    feature_name="churn",
    model_version="1.0",
    platform="VertexAI",
    prediction_dataset_name="my_project",
    prediction_table_name="customers",
    prediction_output_dataset_table="my_project.churn_predictions",
    unique_identifier_column="customer_id",
    predicted_label_column_name="will_churn",
    predicted_probability_column_name="churn_probability",
    optional_configs={"prediction_threshold": 0.5},
)

print(f"\nmodel_name       : {pred_bq.model_name}")
print(f"platform         : {pred_bq.platform}")

print("\n--- Computed fields ---")
print(f"input_source     : {pred_bq.input_source!r}")
# → 'bigquery'
print(f"bigquery_table_ref: {pred_bq.bigquery_table_ref!r}")
# → 'my_project.customers'
print(f"has_output_table : {pred_bq.has_output_table}")
# → True
print(f"is_cloud_prediction: {pred_bq.is_cloud_prediction}")
# → True
print(f"artifact_base_path: {pred_bq.artifact_base_path!r}")
# → 'churn/churn_model/1.0'


# ── 3. PredictionConfig — GCS source ─────────────────────────────────────────

print("\n" + "=" * 60)
print("3. PredictionConfig — GCS source")
print("=" * 60)

pred_gcs = PredictionConfig(
    model_name="churn_model",
    module="churn.predict",
    feature_name="churn",
    model_version="1.0",
    platform="VertexAI",
    input_path="gs://my-bucket/data/customers.parquet",
    output_path="gs://my-bucket/predictions/churn_predictions.parquet",
)

print(f"\ninput_path   : {pred_gcs.input_path}")
print(f"input_source : {pred_gcs.input_source!r}")
# → 'gcs'
print(f"has_output_table: {pred_gcs.has_output_table}")
# → False  (output_path is a file path, not a BQ table)


# ── 4. PredictionConfig — local file source ───────────────────────────────────

print("\n" + "=" * 60)
print("4. PredictionConfig — local file source")
print("=" * 60)

pred_local = PredictionConfig(
    model_name="churn_model",
    module="churn.predict",
    feature_name="churn",
    model_version="dev",
    platform="local",
    input_path="data/sample_inference.csv",
    output_path="artifacts/predictions.csv",
)

print(f"\ninput_path       : {pred_local.input_path}")
print(f"input_source     : {pred_local.input_source!r}")
# → 'local'
print(f"is_cloud_prediction: {pred_local.is_cloud_prediction}")
# → False
print(f"artifact_base_path: {pred_local.artifact_base_path!r}")
# → 'churn/churn_model/dev'
