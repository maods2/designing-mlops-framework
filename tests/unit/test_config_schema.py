"""Unit tests for mlplatform.config.schema."""

from __future__ import annotations

from mlplatform.config.schema import ModelConfig, WorkflowConfig


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
