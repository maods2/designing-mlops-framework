"""Integration tests for config profile merging."""

from __future__ import annotations

import pytest

from mlplatform.config.factory import ConfigLoaderFactory, _load_config_profiles


def _minimal_pipeline_yaml() -> str:
    return (
        "pipeline_name: wf\npipeline_type: training\nfeature_name: f\n"
        "tasks:\n  - task_id: train_model\n    task_type: training\n"
        "    module: example_model.train\n"
    )


class TestConfigFileMerging:
    def test_global_then_dev_override(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: INFO\nbase_path: ./artifacts\n")
        (cfg_dir / "dev.yaml").write_text("log_level: DEBUG\n")

        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        dag = pipeline_dir / "train.yaml"
        dag.write_text(_minimal_pipeline_yaml() + "    config: [global, dev]\n")
        cfg = ConfigLoaderFactory.load_pipeline_config(
            dag, config_names=["global", "dev"], config_dir=cfg_dir
        )
        assert cfg.log_level == "DEBUG"

    def test_missing_config_file_is_skipped(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: INFO\n")

        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        dag = pipeline_dir / "train.yaml"
        dag.write_text(_minimal_pipeline_yaml() + "    config: [global, missing]\n")
        cfg = ConfigLoaderFactory.load_pipeline_config(
            dag, config_names=["global", "missing"], config_dir=cfg_dir
        )
        assert cfg.log_level == "INFO"

    def test_no_config_profiles(self, tmp_path):
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        dag = pipeline_dir / "train.yaml"
        dag.write_text(_minimal_pipeline_yaml())
        cfg = ConfigLoaderFactory.load_pipeline_config(dag)
        assert cfg.config_profiles == []

    def test_example_model_config_dir_discovered(self, train_dag_path):
        """Config dir in example_model/ is auto-discovered from pipeline/train.yaml."""
        cfg = ConfigLoaderFactory.load_pipeline_config(train_dag_path)
        assert cfg.config_profiles

    def test_load_config_profiles_empty(self):
        result = _load_config_profiles([], None)
        assert result == {}

    def test_new_pipeline_task_level_config(self, train_dag_path):
        """New pipeline format reads config: key from task level."""
        cfg = ConfigLoaderFactory.load_pipeline_config(train_dag_path)
        assert "global" in cfg.config_profiles
        assert "dev" in cfg.config_profiles
