"""Pydantic-based configuration models for training, prediction, and pipelines.

These models wrap and extend the underlying dataclass-based schema
(:mod:`mlplatform.config.schema`) with:

* **Input validation** via Pydantic v2
* **Computed fields** that derive useful properties from combinations of
  provided parameters
* **YAML loading** via :meth:`PipelineConfig.from_yaml`

Install
-------
    pip install mlplatform[config]

Usage
-----
Directly::

    from mlplatform.config import TrainingConfig, PredictionConfig, PipelineConfig

    train = TrainingConfig(
        model_name="churn_model",
        module="my_package.train",
        feature_name="churn",
        model_version="1.0",
        platform="VertexAI",
    )
    print(train.artifact_base_path)   # → "churn/churn_model/1.0"
    print(train.is_cloud_training)    # → True

From a YAML DAG file::

    pipeline = PipelineConfig.from_yaml("pipeline/train.yaml")
    for model in pipeline.models:
        print(model.artifact_base_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Configuration for a single training job.

    Computed fields (read-only, derived automatically):

    * ``artifact_base_path`` — storage path prefix for all artifacts belonging
      to this model: ``"{feature_name}/{model_name}/{model_version}"``.
      ``None`` when *feature_name* is not provided.
    * ``is_cloud_training`` — ``True`` when *platform* is not a local variant.
    """

    model_config = ConfigDict(frozen=False, extra="allow")

    model_name: str = Field(..., description="Unique name identifying this model.")
    module: str = Field(..., description="Dotted Python import path to the trainer module.")
    compute: str = Field("s", description="Compute size hint (e.g. 's', 'm', 'l').")
    platform: str = Field("VertexAI", description="Target training platform.")
    model_version: str = Field("latest", description="Artifact version label.")
    feature_name: str | None = Field(
        None, description="Feature domain this model belongs to (set by PipelineConfig)."
    )
    optional_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform key-value pairs passed through to trainer code.",
    )

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def is_cloud_training(self) -> bool:
        """``True`` when the training platform is not a local variant."""
        return self.platform.lower() not in ("local", "local-spark")

    @computed_field  # type: ignore[misc]
    @property
    def artifact_base_path(self) -> str | None:
        """Base artifact path: ``{feature_name}/{model_name}/{model_version}``.

        Returns ``None`` when *feature_name* has not been set.
        """
        if not self.feature_name:
            return None
        return f"{self.feature_name}/{self.model_name}/{self.model_version}"

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        feature_name: str | None = None,
    ) -> "TrainingConfig":
        """Create a ``TrainingConfig`` from an existing :class:`~mlplatform.config.schema.ModelConfig`.

        Args:
            model_config: A :class:`~mlplatform.config.schema.ModelConfig` instance.
            feature_name: Feature domain name (taken from the parent workflow).
        """
        return cls(
            model_name=model_config.model_name,
            module=model_config.module,
            compute=model_config.compute,
            platform=model_config.platform,
            model_version=model_config.model_version,
            optional_configs=model_config.optional_configs,
            feature_name=feature_name,
        )


# ---------------------------------------------------------------------------
# PredictionConfig
# ---------------------------------------------------------------------------


class PredictionConfig(BaseModel):
    """Configuration for a single prediction / inference job.

    Computed fields (read-only, derived automatically):

    * ``input_source`` — detected data source: ``"bigquery"``, ``"gcs"``, or ``"local"``.
    * ``bigquery_table_ref`` — full ``"{dataset}.{table}"`` reference when both
      *prediction_dataset_name* and *prediction_table_name* are provided.
    * ``has_output_table`` — ``True`` when results are written to a BigQuery table.
    * ``is_cloud_prediction`` — ``True`` when *platform* is not a local variant.
    * ``artifact_base_path`` — storage path prefix: ``"{feature_name}/{model_name}/{model_version}"``.
    """

    model_config = ConfigDict(frozen=False, extra="allow")

    model_name: str = Field(..., description="Unique name identifying this model.")
    module: str = Field(..., description="Dotted Python import path to the predictor module.")
    compute: str = Field("s", description="Compute size hint (e.g. 's', 'm', 'l').")
    platform: str = Field("VertexAI", description="Target serving platform.")
    model_version: str = Field("latest", description="Artifact version to load.")
    feature_name: str | None = Field(
        None, description="Feature domain this model belongs to (set by PipelineConfig)."
    )
    optional_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform key-value pairs passed through to predictor code.",
    )

    # prediction-specific
    prediction_dataset_name: str | None = Field(None, description="BigQuery source dataset.")
    prediction_table_name: str | None = Field(None, description="BigQuery source table.")
    model_id: str | None = Field(None, description="Registry model ID (e.g. Vertex AI model ID).")
    prediction_output_dataset_table: str | None = Field(
        None, description="BigQuery destination table for predictions (dataset.table)."
    )
    predicted_label_column_name: str | None = Field(None, description="Output column for predicted label.")
    predicted_timestamp_column_name: str | None = Field(
        None, description="Output column for prediction timestamp."
    )
    predicted_probability_column_name: str | None = Field(
        None, description="Output column for prediction probability."
    )
    unique_identifier_column: str | None = Field(
        None, description="Column used to join predictions back to source rows."
    )
    input_path: str | None = Field(None, description="File path or GCS URI to the input dataset.")
    output_path: str | None = Field(None, description="File path or GCS URI for prediction output.")

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def input_source(self) -> str:
        """Detected data source type.

        Returns:
            ``"bigquery"`` when both dataset and table are configured,
            ``"gcs"`` when *input_path* starts with ``gs://``,
            ``"local"`` otherwise.
        """
        if self.prediction_dataset_name and self.prediction_table_name:
            return "bigquery"
        if self.input_path and self.input_path.startswith("gs://"):
            return "gcs"
        return "local"

    @computed_field  # type: ignore[misc]
    @property
    def bigquery_table_ref(self) -> str | None:
        """Full BigQuery table reference ``"{dataset}.{table}"``, or ``None``."""
        if self.prediction_dataset_name and self.prediction_table_name:
            return f"{self.prediction_dataset_name}.{self.prediction_table_name}"
        return None

    @computed_field  # type: ignore[misc]
    @property
    def has_output_table(self) -> bool:
        """``True`` when prediction results are written to a BigQuery table."""
        return bool(self.prediction_output_dataset_table)

    @computed_field  # type: ignore[misc]
    @property
    def is_cloud_prediction(self) -> bool:
        """``True`` when the serving platform is not a local variant."""
        return self.platform.lower() not in ("local", "local-spark")

    @computed_field  # type: ignore[misc]
    @property
    def artifact_base_path(self) -> str | None:
        """Base artifact path: ``{feature_name}/{model_name}/{model_version}``.

        Returns ``None`` when *feature_name* has not been set.
        """
        if not self.feature_name:
            return None
        return f"{self.feature_name}/{self.model_name}/{self.model_version}"

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        feature_name: str | None = None,
    ) -> "PredictionConfig":
        """Create a ``PredictionConfig`` from an existing :class:`~mlplatform.config.schema.ModelConfig`.

        Args:
            model_config: A :class:`~mlplatform.config.schema.ModelConfig` instance.
            feature_name: Feature domain name (taken from the parent workflow).
        """
        return cls(
            model_name=model_config.model_name,
            module=model_config.module,
            compute=model_config.compute,
            platform=model_config.platform,
            model_version=model_config.model_version,
            optional_configs=model_config.optional_configs,
            feature_name=feature_name,
            prediction_dataset_name=model_config.prediction_dataset_name,
            prediction_table_name=model_config.prediction_table_name,
            model_id=model_config.model_id,
            prediction_output_dataset_table=model_config.prediction_output_dataset_table,
            predicted_label_column_name=model_config.predicted_label_column_name,
            predicted_timestamp_column_name=model_config.predicted_timestamp_column_name,
            predicted_probability_column_name=model_config.predicted_probability_column_name,
            unique_identifier_column=model_config.unique_identifier_column,
            input_path=model_config.input_path,
            output_path=model_config.output_path,
        )


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Full pipeline / workflow configuration.

    The preferred way to create a ``PipelineConfig`` is from a DAG YAML file::

        pipeline = PipelineConfig.from_yaml("pipeline/train.yaml")

    Computed fields (read-only, derived automatically):

    * ``is_training`` — ``True`` when *pipeline_type* is ``"training"``.
    * ``is_prediction`` — ``True`` when *pipeline_type* is ``"prediction"``.
    * ``model_count`` — number of model configurations in the pipeline.
    """

    model_config = ConfigDict(frozen=False)

    workflow_name: str = Field(..., description="Human-readable name for this pipeline.")
    execution_mode: str = Field("sequential", description="How tasks are scheduled (e.g. 'sequential').")
    pipeline_type: Literal["training", "prediction"] = Field(
        "training", description="Whether this pipeline trains or runs inference."
    )
    feature_name: str = Field(..., description="Feature domain name used to namespace artifacts.")
    config_version: int = Field(2, description="DAG template format version.")
    log_level: str = Field("INFO", description="Logging verbosity level.")
    models: list[Union[TrainingConfig, PredictionConfig]] = Field(
        default_factory=list, description="Per-model configurations in pipeline order."
    )
    config_profiles: list[str] = Field(
        default_factory=list, description="Config profile names that were merged to produce this config."
    )

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def is_training(self) -> bool:
        """``True`` when this is a training pipeline."""
        return self.pipeline_type == "training"

    @computed_field  # type: ignore[misc]
    @property
    def is_prediction(self) -> bool:
        """``True`` when this is a prediction / inference pipeline."""
        return self.pipeline_type == "prediction"

    @computed_field  # type: ignore[misc]
    @property
    def model_count(self) -> int:
        """Number of model configurations in this pipeline."""
        return len(self.models)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        dag_path: str | Path,
        config_names: list[str] | None = None,
        config_dir: str | Path | None = None,
    ) -> "PipelineConfig":
        """Load and validate a ``PipelineConfig`` from a DAG YAML file.

        Wraps :func:`~mlplatform.config.loader.load_workflow_config` and
        converts the resulting :class:`~mlplatform.config.schema.WorkflowConfig`
        into a fully validated ``PipelineConfig`` with Pydantic.

        Args:
            dag_path: Path to the DAG YAML file.
            config_names: Override every task's ``config:`` key (e.g.
                ``["global", "local"]``).  ``None`` uses each task's own key.
            config_dir: Directory containing config profile YAML files.
                Defaults to auto-detection (searches for ``config/`` near the
                DAG file).

        Returns:
            A validated :class:`PipelineConfig` instance.

        Example::

            pipeline = PipelineConfig.from_yaml(
                "example_model/pipeline/train.yaml",
                config_names=["global", "dev"],
            )
            print(pipeline.is_training)   # True
            print(pipeline.model_count)   # 1
        """
        from mlplatform.config.loader import load_workflow_config

        workflow = load_workflow_config(
            dag_path=dag_path,
            config_names=config_names,
            config_dir=config_dir,
        )

        feature_name = workflow.feature_name
        pipeline_type = workflow.pipeline_type

        typed_models: list[TrainingConfig | PredictionConfig] = []
        for mc in workflow.models:
            if pipeline_type == "training":
                typed_models.append(TrainingConfig.from_model_config(mc, feature_name=feature_name))
            else:
                typed_models.append(PredictionConfig.from_model_config(mc, feature_name=feature_name))

        return cls(
            workflow_name=workflow.workflow_name,
            execution_mode=workflow.execution_mode,
            pipeline_type=pipeline_type,
            feature_name=feature_name,
            config_version=workflow.config_version,
            log_level=workflow.log_level,
            models=typed_models,
            config_profiles=workflow.config_profiles,
        )
