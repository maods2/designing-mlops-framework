"""Integration tests for config profile merging."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlplatform.config.loader import load_workflow_config, _deep_merge, _load_config_profiles


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
        # "missing.yaml" does not exist — should not raise

        dag = tmp_path / "train.yaml"
        dag.write_text(
            "workflow_name: wf\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: f\nconfig_version: 1\nmodels: []\n"
        )

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

    def test_project_root_config_dir_discovered(self, train_dag_path, repo_root):
        """Config dir at project root is auto-discovered."""
        config_dir = repo_root / "config"
        assert config_dir.is_dir(), "config/ should exist at project root"
        cfg = load_workflow_config(train_dag_path)
        # global.yaml and dev.yaml exist in config/, so profiles should be populated
        assert cfg.config_profiles

    def test_load_config_profiles_order(self):
        """_load_config_profiles returns an empty dict for empty profile list."""
        result = _load_config_profiles([], None)
        assert result == {}
