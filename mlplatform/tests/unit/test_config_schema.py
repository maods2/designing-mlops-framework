"""Unit tests for mlplatform.config.schema."""

from __future__ import annotations

from mlplatform.config.schema import (
    ModelConfig,
    TaskConfig,
    UnifiedPipelineConfig,
    WorkflowConfig,
)


class TestTaskConfig:
    def test_to_model_config(self):
        task = TaskConfig(
            task_id="train_model",
            task_type="training",
            model_name="my_model",
            module="pkg.train",
        )
        model = task.to_model_config()
        assert model.model_name == "my_model"
        assert model.module == "pkg.train"
        assert model.compute == "s"

    def test_to_model_config_fallback_model_name(self):
        task = TaskConfig(task_id="predict", task_type="prediction", module="pkg.predict")
        model = task.to_model_config()
        assert model.model_name == "predict"


class TestUnifiedPipelineConfig:
    def test_defaults(self):
        cfg = UnifiedPipelineConfig(
            pipeline_name="wf",
            pipeline_type="training",
            feature_name="feat",
        )
        assert cfg.schedule == {}
        assert cfg.environments == {}
        assert cfg.tasks == []
        assert cfg.log_level == "INFO"


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
