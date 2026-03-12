"""Unit tests for mlplatform.config models.

Covers TrainingConfig, PredictionConfig, and RunConfig.
"""

from __future__ import annotations

import pytest

from mlplatform.config import PredictionConfig, RunConfig, TrainingConfig


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_from_dict(self):
        kwargs = {"model_name": "m", "feature": "churn", "version": "v1"}
        cfg = TrainingConfig(kwargs)
        assert cfg.model_name == "m"
        assert cfg.feature == "churn"
        assert cfg.version == "v1"
        assert cfg.artifact_base_path == "churn/m/v1"

    def test_from_kwargs(self):
        cfg = TrainingConfig(model_name="m", feature="churn", version="v1")
        assert cfg.model_name == "m"
        assert cfg.feature == "churn"

    def test_normalize_feature_name_alias(self):
        kwargs = {"model_name": "m", "feature_name": "sales", "version": "v1"}
        cfg = TrainingConfig(kwargs)
        assert cfg.feature == "sales"

    def test_normalize_model_version_alias(self):
        kwargs = {"model_name": "m", "feature": "f", "model_version": "2.0"}
        cfg = TrainingConfig(kwargs)
        assert cfg.version == "2.0"

    def test_is_cloud_training_true(self):
        cfg = TrainingConfig(model_name="m", feature="f", platform="VertexAI")
        assert cfg.is_cloud_training is True

    def test_is_cloud_training_false_for_local(self):
        cfg = TrainingConfig(model_name="m", feature="f", platform="local")
        assert cfg.is_cloud_training is False

    def test_to_artifact_kwargs(self):
        cfg = TrainingConfig(
            model_name="m",
            feature="f",
            version="v1",
            base_path="/out",
        )
        kw = cfg.to_artifact_kwargs()
        assert kw["model_name"] == "m"
        assert kw["feature"] == "f"
        assert kw["version"] == "v1"
        assert kw["base_path"] == "/out"

    def test_missing_required_raises(self):
        with pytest.raises(Exception):
            TrainingConfig(model_name="m")  # feature required
        with pytest.raises(Exception):
            TrainingConfig(feature="f")  # model_name required


# ---------------------------------------------------------------------------
# PredictionConfig
# ---------------------------------------------------------------------------


class TestPredictionConfig:
    def test_from_dict(self):
        kwargs = {"model_name": "m", "feature": "churn", "version": "v1"}
        cfg = PredictionConfig(kwargs)
        assert cfg.model_name == "m"
        assert cfg.input_source == "local"

    def test_input_source_bigquery(self):
        cfg = PredictionConfig(
            model_name="m",
            feature="f",
            prediction_dataset_name="ds",
            prediction_table_name="tbl",
        )
        assert cfg.input_source == "bigquery"
        assert cfg.bigquery_table_ref == "ds.tbl"

    def test_input_source_gcs(self):
        cfg = PredictionConfig(
            model_name="m",
            feature="f",
            input_path="gs://bucket/in.csv",
        )
        assert cfg.input_source == "gcs"

    def test_has_output_table(self):
        cfg = PredictionConfig(
            model_name="m",
            feature="f",
            prediction_output_dataset_table="ds.out",
        )
        assert cfg.has_output_table is True

    def test_is_cloud_prediction_false_for_local(self):
        cfg = PredictionConfig(model_name="m", feature="f", platform="local")
        assert cfg.is_cloud_prediction is False

    def test_artifact_base_path(self):
        cfg = PredictionConfig(
            model_name="fraud_model",
            feature="fraud",
            version="1.2",
        )
        assert cfg.artifact_base_path == "fraud/fraud_model/1.2"

    def test_to_artifact_kwargs(self):
        cfg = PredictionConfig(model_name="m", feature="f", version="v1")
        kw = cfg.to_artifact_kwargs()
        assert kw["model_name"] == "m"
        assert kw["feature"] == "f"


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_to_artifact_kwargs(self):
        cfg = RunConfig(model_name="m", feature="f", version="v1", base_path="/out")
        kw = cfg.to_artifact_kwargs()
        assert kw["model_name"] == "m"
        assert kw["feature"] == "f"
        assert kw["version"] == "v1"
        assert kw["base_path"] == "/out"
        assert kw["backend"] == "local"

    def test_defaults(self):
        cfg = RunConfig(model_name="m", feature="f")
        assert cfg.version == "dev"
        assert cfg.backend == "local"
        assert cfg.base_path == "./artifacts"


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_is_importable(self):
        from mlplatform import __version__

        assert isinstance(__version__, str)
        assert __version__.count(".") >= 2

    def test_version_matches_version_file(self):
        from mlplatform import __version__
        from mlplatform._version import __version__ as ver_file

        assert __version__ == ver_file
