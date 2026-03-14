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


class PipelineConfig(BaseModel):
    """Frozen, validated configuration for a single model execution.

    The single config model passed everywhere — replaces scattered args.
    Construct via :class:`PipelineConfigBuilder`, :meth:`from_dict`, or directly.

    Example::

        config = PipelineConfig(
            model_name="churn_model",
            feature="churn",
            pipeline_type="training",
        )
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    model_name: str = Field(..., description="Model identifier.")
    feature: str = Field(..., description="Feature domain (e.g. churn, demo).")
    version: str = Field("dev", description="Artifact version.")

    # Infrastructure
    base_path: str = Field("./artifacts", description="Local base path for artifacts.")
    base_bucket: str | None = Field(None, description="GCS bucket or local path override.")
    backend: Literal["local", "gcs"] = Field("local", description="Storage backend.")
    project_id: str | None = Field(None, description="GCP project (for gcs backend).")

    # Pipeline
    pipeline_type: Literal["training", "prediction"] = Field(
        "training", description="Whether this executes training or prediction."
    )
    profile: str = Field("local", description="Infrastructure profile name.")
    platform: str = Field("VertexAI", description="Target platform.")

    # Module resolution
    module: str = Field("", description="Dotted import path, e.g. 'model_code.train:MyTrainer'.")

    # Config loading
    config_list: list[str] = Field(
        default_factory=lambda: ["global", "dev"],
        description="Config profile YAML names to merge.",
    )
    config_dir: str = Field("./config", description="Directory containing config profile YAMLs.")

    # Merged user config (from YAML profiles or direct dict)
    user_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged config from YAML profiles or orchestrator overrides.",
    )

    # Metadata
    commit_hash: str | None = Field(None, description="Git commit hash for reproducibility.")
    log_level: str = Field("INFO", description="Logging verbosity level.")

    # Prediction-specific (optional, for InferenceStrategy compatibility)
    input_path: str | None = Field(None, description="File path or GCS URI for prediction input.")
    output_path: str | None = Field(None, description="File path or GCS URI for prediction output.")
    prediction_dataset_name: str | None = Field(None, description="BigQuery source dataset.")
    prediction_table_name: str | None = Field(None, description="BigQuery source table.")
    prediction_output_dataset_table: str | None = Field(
        None, description="BigQuery destination table (dataset.table)."
    )

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        """Construct from a flat dict (e.g. orchestrator JSON payload).

        If ``config_list`` and ``config_dir`` are present, loads and merges
        YAML profiles into ``user_config`` before constructing the frozen model.
        """
        from mlplatform.config.loader import load_config_profiles

        data = dict(payload)
        config_list = data.pop("config_list", None)
        config_dir = data.pop("config_dir", "./config")

        if config_list:
            merged = load_config_profiles(config_list, config_dir)
            # Identity / infra fields in YAML feed into top-level fields
            _PROMOTE_KEYS = (
                "model_name", "feature", "version", "base_path",
                "base_bucket", "backend", "project_id", "profile",
                "log_level", "platform", "module", "input_path",
                "output_path", "prediction_dataset_name",
                "prediction_table_name", "prediction_output_dataset_table",
            )
            for key in _PROMOTE_KEYS:
                if key in merged:
                    if key not in data:
                        data[key] = merged[key]
                    del merged[key]
            existing_uc = data.get("user_config", {})
            data["user_config"] = {**merged, **existing_uc}

        data.setdefault("config_list", config_list or ["global", "dev"])
        data.setdefault("config_dir", config_dir)
        return cls(**data)
