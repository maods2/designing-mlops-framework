"""Pydantic-based configuration models for training and prediction.

* **TrainingConfig** — validated config for training; accepts kwargs dict.
* **PredictionConfig** — validated config for prediction; accepts kwargs dict.
* **RunConfig** — config for train/predict artifact runs.

Usage
-----
From kwargs dict::

    def train(kwargs):
        cfg = TrainingConfig(kwargs)
        artifact = Artifact(**cfg.to_artifact_kwargs())

    def predict(kwargs):
        cfg = PredictionConfig(kwargs)

Or with keyword args::

    cfg = TrainingConfig(model_name="m", feature="churn", version="v1")
    artifact = Artifact(**cfg.to_artifact_kwargs())
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


def _normalize_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize aliases: feature_name->feature, model_version->version."""
    out = dict(data)
    if "feature_name" in out and "feature" not in out:
        out["feature"] = out.pop("feature_name")
    if "model_version" in out and "version" not in out:
        out["version"] = out.pop("model_version")
    return out


class TrainingConfig(BaseModel):
    """Configuration for a single training job.

    Accepts a kwargs dict or keyword args. Use :meth:`to_artifact_kwargs` for
    :class:`~mlplatform.artifacts.Artifact`.
    """

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(..., description="Model identifier.")
    feature: str = Field(..., description="Feature domain (e.g. churn, demo).")
    version: str = Field("dev", description="Artifact version.")
    base_path: str = Field("./artifacts", description="Local base path for artifacts.")
    base_bucket: str | None = Field(None, description="GCS bucket or local path override.")
    backend: Literal["local", "gcs"] = Field("local", description="Storage backend.")
    project_id: str | None = Field(None, description="GCP project (for gcs backend).")
    platform: str = Field("VertexAI", description="Target platform.")
    optional_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform key-value pairs passed through to trainer.",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            super().__init__(**_normalize_kwargs(args[0]))
        else:
            super().__init__(**_normalize_kwargs(kwargs))

    @computed_field  # type: ignore[misc]
    @property
    def artifact_base_path(self) -> str:
        """Base artifact path: ``{feature}/{model_name}/{version}``."""
        return f"{self.feature}/{self.model_name}/{self.version}"

    @computed_field  # type: ignore[misc]
    @property
    def is_cloud_training(self) -> bool:
        """``True`` when platform is not a local variant."""
        return self.platform.lower() not in ("local", "local-spark")

    def to_artifact_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for Artifact()."""
        return {
            "model_name": self.model_name,
            "feature": self.feature,
            "version": self.version,
            "base_path": self.base_path,
            "base_bucket": self.base_bucket,
            "backend": self.backend,
            "project_id": self.project_id,
        }


class PredictionConfig(BaseModel):
    """Configuration for a single prediction / inference job.

    Accepts a kwargs dict or keyword args. Use :meth:`to_artifact_kwargs` for
    :class:`~mlplatform.artifacts.Artifact`.
    """

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(..., description="Model identifier.")
    feature: str = Field(..., description="Feature domain (e.g. churn, demo).")
    version: str = Field("dev", description="Artifact version.")
    base_path: str = Field("./artifacts", description="Local base path for artifacts.")
    base_bucket: str | None = Field(None, description="GCS bucket or local path override.")
    backend: Literal["local", "gcs"] = Field("local", description="Storage backend.")
    project_id: str | None = Field(None, description="GCP project (for gcs backend).")
    platform: str = Field("VertexAI", description="Target platform.")
    optional_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform key-value pairs passed through to predictor.",
    )

    # prediction-specific
    prediction_dataset_name: str | None = Field(None, description="BigQuery source dataset.")
    prediction_table_name: str | None = Field(None, description="BigQuery source table.")
    prediction_output_dataset_table: str | None = Field(
        None, description="BigQuery destination table (dataset.table)."
    )
    model_id: str | None = Field(None, description="Registry model ID.")
    predicted_label_column_name: str | None = Field(None, description="Output column for label.")
    predicted_timestamp_column_name: str | None = Field(None, description="Output column for timestamp.")
    predicted_probability_column_name: str | None = Field(None, description="Output column for probability.")
    unique_identifier_column: str | None = Field(None, description="Column to join predictions.")
    input_path: str | None = Field(None, description="File path or GCS URI for input.")
    output_path: str | None = Field(None, description="File path or GCS URI for output.")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            super().__init__(**_normalize_kwargs(args[0]))
        else:
            super().__init__(**_normalize_kwargs(kwargs))

    @computed_field  # type: ignore[misc]
    @property
    def artifact_base_path(self) -> str:
        """Base artifact path: ``{feature}/{model_name}/{version}``."""
        return f"{self.feature}/{self.model_name}/{self.version}"

    @computed_field  # type: ignore[misc]
    @property
    def input_source(self) -> str:
        """Detected data source: ``bigquery``, ``gcs``, or ``local``."""
        if self.prediction_dataset_name and self.prediction_table_name:
            return "bigquery"
        if self.input_path and self.input_path.startswith("gs://"):
            return "gcs"
        return "local"

    @computed_field  # type: ignore[misc]
    @property
    def bigquery_table_ref(self) -> str | None:
        """Full BigQuery table reference ``{dataset}.{table}``, or ``None``."""
        if self.prediction_dataset_name and self.prediction_table_name:
            return f"{self.prediction_dataset_name}.{self.prediction_table_name}"
        return None

    @computed_field  # type: ignore[misc]
    @property
    def has_output_table(self) -> bool:
        """``True`` when results are written to a BigQuery table."""
        return bool(self.prediction_output_dataset_table)

    @computed_field  # type: ignore[misc]
    @property
    def is_cloud_prediction(self) -> bool:
        """``True`` when platform is not a local variant."""
        return self.platform.lower() not in ("local", "local-spark")

    def to_artifact_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for Artifact()."""
        return {
            "model_name": self.model_name,
            "feature": self.feature,
            "version": self.version,
            "base_path": self.base_path,
            "base_bucket": self.base_bucket,
            "backend": self.backend,
            "project_id": self.project_id,
        }


class RunConfig(BaseModel):
    """Config for train and predict artifact runs.

    Wraps kwargs passed to :class:`~mlplatform.artifacts.Artifact`.
    """

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(..., description="Model identifier.")
    feature: str = Field(..., description="Feature domain (e.g. churn, demo).")
    version: str = Field("dev", description="Artifact version.")
    base_path: str = Field("./artifacts", description="Local base path for artifacts.")
    base_bucket: str | None = Field(None, description="GCS bucket (when backend=gcs) or local path override.")
    backend: Literal["local", "gcs"] = Field("local", description="Storage backend.")
    project_id: str | None = Field(None, description="GCP project (for gcs backend).")

    def to_artifact_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for Artifact()."""
        return {
            "model_name": self.model_name,
            "feature": self.feature,
            "version": self.version,
            "base_path": self.base_path,
            "base_bucket": self.base_bucket,
            "backend": self.backend,
            "project_id": self.project_id,
        }
