"""Unit tests for mlplatform.config Pydantic models.

Covers ModelConfig, WorkflowConfig, TrainingConfig, PredictionConfig, and
PipelineConfig including computed fields, YAML loading, and validation behaviour.
"""

from __future__ import annotations

import pytest

from mlplatform.config.models import (
    ModelConfig,
    PipelineConfig,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(**kwargs) -> ModelConfig:
    defaults = dict(
        model_name="test_model",
        module="my_pkg.train",
        compute="s",
        platform="VertexAI",
        model_version="1.0",
        optional_configs={},
        prediction_dataset_name=None,
        prediction_table_name=None,
        model_id=None,
        prediction_output_dataset_table=None,
        predicted_label_column_name=None,
        predicted_timestamp_column_name=None,
        predicted_probability_column_name=None,
        unique_identifier_column=None,
        input_path=None,
        output_path=None,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# ModelConfig / WorkflowConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model_name="m", module="m.train")
        assert cfg.compute == "s"
        assert cfg.platform == "VertexAI"
        assert cfg.optional_configs == {}
        assert cfg.model_version == "latest"
        assert cfg.input_path is None
        assert cfg.output_path is None

    def test_full_config(self):
        cfg = ModelConfig(
            model_name="my_model",
            module="pkg.train",
            compute="m",
            platform="Dataproc",
            optional_configs={"threshold": 0.5},
            input_path="/data/in.csv",
            output_path="/data/out.parquet",
            model_version="v1",
        )
        assert cfg.compute == "m"
        assert cfg.optional_configs["threshold"] == 0.5
        assert cfg.input_path == "/data/in.csv"


class TestWorkflowConfig:
    def test_defaults(self):
        cfg = WorkflowConfig(
            workflow_name="wf",
            execution_mode="sequential",
            pipeline_type="training",
            feature_name="feat",
            config_version=2,
            models=[],
        )
        assert cfg.log_level == "INFO"
        assert cfg.config_profiles == []

    def test_config_profiles_field(self):
        cfg = WorkflowConfig(
            workflow_name="wf",
            execution_mode="sequential",
            pipeline_type="training",
            feature_name="feat",
            config_version=2,
            models=[],
            config_profiles=["global", "dev"],
        )
        assert cfg.config_profiles == ["global", "dev"]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_basic_construction(self):
        cfg = TrainingConfig(model_name="m", module="pkg.train")
        assert cfg.model_name == "m"
        assert cfg.compute == "s"
        assert cfg.platform == "VertexAI"
        assert cfg.model_version == "latest"

    def test_is_cloud_training_true_for_vertexai(self):
        cfg = TrainingConfig(model_name="m", module="pkg.train", platform="VertexAI")
        assert cfg.is_cloud_training is True

    def test_is_cloud_training_false_for_local(self):
        cfg = TrainingConfig(model_name="m", module="pkg.train", platform="local")
        assert cfg.is_cloud_training is False

    def test_is_cloud_training_false_for_local_spark(self):
        cfg = TrainingConfig(model_name="m", module="pkg.train", platform="local-spark")
        assert cfg.is_cloud_training is False

    def test_artifact_base_path_with_feature_name(self):
        cfg = TrainingConfig(
            model_name="churn_model",
            module="pkg.train",
            feature_name="churn",
            model_version="2.0",
        )
        assert cfg.artifact_base_path == "churn/churn_model/2.0"

    def test_artifact_base_path_none_without_feature_name(self):
        cfg = TrainingConfig(model_name="m", module="pkg.train")
        assert cfg.artifact_base_path is None

    def test_from_model_config(self):
        mc = _make_model_config(
            model_name="sales_model",
            module="sales.train",
            model_version="v3",
            platform="Databricks",
        )
        cfg = TrainingConfig.from_model_config(mc, feature_name="sales")
        assert cfg.model_name == "sales_model"
        assert cfg.module == "sales.train"
        assert cfg.model_version == "v3"
        assert cfg.feature_name == "sales"
        assert cfg.artifact_base_path == "sales/sales_model/v3"

    def test_from_model_config_no_feature_name(self):
        mc = _make_model_config()
        cfg = TrainingConfig.from_model_config(mc)
        assert cfg.feature_name is None
        assert cfg.artifact_base_path is None

    def test_optional_configs_passed_through(self):
        mc = _make_model_config(optional_configs={"learning_rate": 0.01, "epochs": 50})
        cfg = TrainingConfig.from_model_config(mc)
        assert cfg.optional_configs["learning_rate"] == 0.01

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            TrainingConfig(module="pkg.train")  # model_name is required


# ---------------------------------------------------------------------------
# PredictionConfig
# ---------------------------------------------------------------------------


class TestPredictionConfig:
    def test_basic_construction(self):
        cfg = PredictionConfig(model_name="m", module="pkg.predict")
        assert cfg.model_version == "latest"
        assert cfg.input_source == "local"
        assert cfg.bigquery_table_ref is None
        assert cfg.has_output_table is False
        assert cfg.is_cloud_prediction is True

    def test_input_source_bigquery(self):
        cfg = PredictionConfig(
            model_name="m",
            module="pkg.predict",
            prediction_dataset_name="my_dataset",
            prediction_table_name="my_table",
        )
        assert cfg.input_source == "bigquery"
        assert cfg.bigquery_table_ref == "my_dataset.my_table"

    def test_input_source_gcs(self):
        cfg = PredictionConfig(
            model_name="m",
            module="pkg.predict",
            input_path="gs://my-bucket/data/input.csv",
        )
        assert cfg.input_source == "gcs"
        assert cfg.bigquery_table_ref is None

    def test_input_source_local_file(self):
        cfg = PredictionConfig(
            model_name="m",
            module="pkg.predict",
            input_path="/data/local.csv",
        )
        assert cfg.input_source == "local"

    def test_bigquery_table_ref_none_when_only_dataset(self):
        cfg = PredictionConfig(
            model_name="m",
            module="pkg.predict",
            prediction_dataset_name="ds",
        )
        assert cfg.bigquery_table_ref is None
        assert cfg.input_source == "local"

    def test_has_output_table_true(self):
        cfg = PredictionConfig(
            model_name="m",
            module="pkg.predict",
            prediction_output_dataset_table="ds.output_table",
        )
        assert cfg.has_output_table is True

    def test_is_cloud_prediction_false_for_local(self):
        cfg = PredictionConfig(model_name="m", module="pkg.predict", platform="local")
        assert cfg.is_cloud_prediction is False

    def test_artifact_base_path(self):
        cfg = PredictionConfig(
            model_name="fraud_model",
            module="fraud.predict",
            feature_name="fraud",
            model_version="1.2",
        )
        assert cfg.artifact_base_path == "fraud/fraud_model/1.2"

    def test_from_model_config_full(self):
        mc = _make_model_config(
            model_name="pred_model",
            module="pkg.predict",
            prediction_dataset_name="ds",
            prediction_table_name="tbl",
            model_version="2.0",
            unique_identifier_column="user_id",
        )
        cfg = PredictionConfig.from_model_config(mc, feature_name="churn")
        assert cfg.model_name == "pred_model"
        assert cfg.bigquery_table_ref == "ds.tbl"
        assert cfg.input_source == "bigquery"
        assert cfg.feature_name == "churn"
        assert cfg.artifact_base_path == "churn/pred_model/2.0"
        assert cfg.unique_identifier_column == "user_id"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def _training_pipeline(self, **kwargs) -> PipelineConfig:
        defaults = dict(
            workflow_name="train_pipeline",
            pipeline_type="training",
            feature_name="churn",
            models=[
                TrainingConfig(model_name="m1", module="pkg.train", feature_name="churn")
            ],
        )
        defaults.update(kwargs)
        return PipelineConfig(**defaults)

    def test_is_training_true(self):
        p = self._training_pipeline()
        assert p.is_training is True
        assert p.is_prediction is False

    def test_is_prediction_true(self):
        p = PipelineConfig(
            workflow_name="pred_pipeline",
            pipeline_type="prediction",
            feature_name="churn",
            models=[
                PredictionConfig(model_name="m1", module="pkg.predict", feature_name="churn")
            ],
        )
        assert p.is_prediction is True
        assert p.is_training is False

    def test_model_count(self):
        p = self._training_pipeline()
        assert p.model_count == 1

    def test_model_count_multiple(self):
        p = PipelineConfig(
            workflow_name="multi",
            pipeline_type="training",
            feature_name="f",
            models=[
                TrainingConfig(model_name="m1", module="a.train"),
                TrainingConfig(model_name="m2", module="b.train"),
            ],
        )
        assert p.model_count == 2

    def test_empty_pipeline(self):
        p = PipelineConfig(
            workflow_name="empty",
            pipeline_type="training",
            feature_name="f",
        )
        assert p.model_count == 0

    def test_invalid_pipeline_type_raises(self):
        with pytest.raises(Exception):
            PipelineConfig(
                workflow_name="bad",
                pipeline_type="scoring",  # not a valid Literal
                feature_name="f",
            )

    def test_from_yaml_training(self, tmp_path):
        dag = tmp_path / "train.yaml"
        dag.write_text(
            """
workflow_name: test_train
execution_mode: sequential
pipeline_type: training
feature_name: churn
config_version: 2
log_level: INFO
models:
  - model_name: churn_model
    module: churn.train
    compute: s
    training_platform: VertexAI
    model_version: "1.0"
"""
        )
        pipeline = PipelineConfig.from_yaml(dag)

        assert pipeline.workflow_name == "test_train"
        assert pipeline.is_training is True
        assert pipeline.feature_name == "churn"
        assert pipeline.model_count == 1

        model = pipeline.models[0]
        assert isinstance(model, TrainingConfig)
        assert model.model_name == "churn_model"
        assert model.feature_name == "churn"
        assert model.artifact_base_path == "churn/churn_model/1.0"
        assert model.is_cloud_training is True

    def test_from_yaml_prediction(self, tmp_path):
        dag = tmp_path / "predict.yaml"
        dag.write_text(
            """
workflow_name: test_predict
execution_mode: sequential
pipeline_type: prediction
feature_name: fraud
config_version: 2
models:
  - model_name: fraud_model
    module: fraud.predict
    compute: m
    serving_platform: VertexAI
    model_version: "2.0"
    prediction_dataset_name: my_ds
    prediction_table_name: my_tbl
    prediction_output_dataset_table: out_ds.predictions
"""
        )
        pipeline = PipelineConfig.from_yaml(dag)

        assert pipeline.is_prediction is True
        model = pipeline.models[0]
        assert isinstance(model, PredictionConfig)
        assert model.input_source == "bigquery"
        assert model.bigquery_table_ref == "my_ds.my_tbl"
        assert model.has_output_table is True
        assert model.artifact_base_path == "fraud/fraud_model/2.0"

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_is_importable(self):
        from mlplatform import __version__

        assert isinstance(__version__, str)
        assert __version__.count(".") == 2  # semver X.Y.Z

    def test_version_matches_version_file(self):
        from mlplatform import __version__
        from mlplatform._version import __version__ as ver_file

        assert __version__ == ver_file
