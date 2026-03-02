"""Unit tests for mlplatform.config.factory (new schema loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlplatform.config.factory import ConfigLoaderFactory, _deep_merge


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 99, "c": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}}

    def test_empty_override(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}

    def test_empty_base(self):
        override = {"a": 1}
        assert _deep_merge({}, override) == {"a": 1}


class TestLoadPipelineConfig:
    def test_load_train_pipeline(self, train_dag_path):
        cfg = ConfigLoaderFactory.load_pipeline_config(train_dag_path)
        assert cfg.pipeline_name == "example_workflow_sequential"
        assert cfg.pipeline_type == "training"
        assert cfg.feature_name == "example"
        assert len(cfg.tasks) >= 1
        train_task = next(t for t in cfg.tasks if t.task_id == "train_model")
        assert train_task.module == "example_model.train"

    def test_load_predict_pipeline(self, predict_dag_path):
        cfg = ConfigLoaderFactory.load_pipeline_config(predict_dag_path)
        assert cfg.pipeline_type == "prediction"
        predict_task = next(t for t in cfg.tasks if t.task_id == "predict")
        assert predict_task.input_path is not None or predict_task.module

    def test_load_single_task(self, train_dag_path):
        cfg = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, task_id="train_model"
        )
        assert len(cfg.tasks) == 1
        assert cfg.tasks[0].task_id == "train_model"

    def test_config_profiles_from_task(self, train_dag_path):
        cfg = ConfigLoaderFactory.load_pipeline_config(train_dag_path)
        assert "global" in cfg.config_profiles
        assert "dev" in cfg.config_profiles

    def test_config_names_cli_override(self, train_dag_path):
        cfg = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, config_names=["global", "train-local"]
        )
        assert "global" in cfg.config_profiles
        assert "train-local" in cfg.config_profiles

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ConfigLoaderFactory.load_pipeline_config(tmp_path / "nonexistent.yaml")

    def test_config_profile_merging(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: INFO\n")
        (cfg_dir / "train-local.yaml").write_text("log_level: DEBUG\n")

        pipeline_file = tmp_path / "pipeline" / "train.yaml"
        pipeline_file.parent.mkdir(parents=True)
        pipeline_file.write_text(
            "pipeline_name: test\npipeline_type: training\nfeature_name: test\n"
            "tasks:\n  - task_id: train_model\n    task_type: training\n"
            "    config: [global, train-local]\n    module: example_model.train\n"
        )
        cfg = ConfigLoaderFactory.load_pipeline_config(
            pipeline_file, config_dir=cfg_dir
        )
        assert cfg.log_level == "DEBUG"

    def test_task_not_found_raises(self, train_dag_path):
        with pytest.raises(ValueError, match="Task 'nonexistent' not found"):
            ConfigLoaderFactory.load_pipeline_config(
                train_dag_path, task_id="nonexistent"
            )
