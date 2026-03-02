"""Integration tests for config profile merging."""

from __future__ import annotations

import pytest

from mlplatform.config.loader import load_workflow_config, _load_config_profiles


class TestConfigFileMerging:
    def test_global_then_dev_override(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: INFO\nbase_path: ./artifacts\n")
        (cfg_dir / "dev.yaml").write_text("log_level: DEBUG\n")

        dag = tmp_path / "train.yaml"
        dag.write_text(
            "workflow_name: wf\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: f\nconfig_version: 1\nmodels: []\n"
        )
        cfg = load_workflow_config(dag, config_names=["global", "dev"], config_dir=cfg_dir)
        assert cfg.log_level == "DEBUG"

    def test_missing_config_file_is_skipped(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: INFO\n")

        dag = tmp_path / "train.yaml"
        dag.write_text(
            "workflow_name: wf\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: f\nconfig_version: 1\nmodels: []\n"
        )
        # "missing.yaml" does not exist — should not raise
        cfg = load_workflow_config(dag, config_names=["global", "missing"], config_dir=cfg_dir)
        assert cfg.log_level == "INFO"

    def test_no_config_profiles(self, tmp_path):
        dag = tmp_path / "train.yaml"
        dag.write_text(
            "workflow_name: wf\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: f\nconfig_version: 1\nmodels: []\n"
        )
        cfg = load_workflow_config(dag)
        assert cfg.config_profiles == []

    def test_example_model_config_dir_discovered(self, train_dag_path):
        """Config dir in example_model/ is auto-discovered from pipeline/train.yaml."""
        cfg = load_workflow_config(train_dag_path)
        # global.yaml and dev.yaml exist in example_model/config/
        assert cfg.config_profiles  # non-empty

    def test_load_config_profiles_empty(self):
        result = _load_config_profiles([], None)
        assert result == {}

    def test_new_dag_task_level_config(self, train_dag_path):
        """New DAG format reads config: key from task level."""
        cfg = load_workflow_config(train_dag_path)
        assert "global" in cfg.config_profiles
        assert "dev" in cfg.config_profiles
