"""Unit tests for mlplatform.config.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlplatform.config.loader import load_workflow_config, _deep_merge, _load_config_profiles


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


class TestLoadWorkflowConfig:
    def test_load_legacy_train_dag(self, legacy_train_dag_path):
        cfg = load_workflow_config(legacy_train_dag_path)
        assert cfg.workflow_name == "eds_workflow_sequential"
        assert cfg.pipeline_type == "training"
        assert cfg.feature_name == "eds"
        assert len(cfg.models) == 1
        assert cfg.models[0].model_name == "lr_p708"

    def test_load_legacy_predict_dag(self, legacy_predict_dag_path):
        cfg = load_workflow_config(legacy_predict_dag_path)
        assert cfg.pipeline_type == "prediction"
        assert cfg.models[0].input_path is not None

    def test_load_new_train_dag(self, train_dag_path):
        cfg = load_workflow_config(train_dag_path)
        assert cfg.workflow_name == "example_workflow_sequential"
        assert cfg.pipeline_type == "training"
        assert cfg.feature_name == "example"
        assert len(cfg.models) == 1
        assert cfg.models[0].module == "example_model.train"

    def test_load_new_predict_dag(self, predict_dag_path):
        cfg = load_workflow_config(predict_dag_path)
        assert cfg.pipeline_type == "prediction"
        assert cfg.models[0].input_path is not None

    def test_config_profiles_loaded(self, train_dag_path):
        """DAG declares config: [global, dev] — profiles should be recorded."""
        cfg = load_workflow_config(train_dag_path)
        assert "global" in cfg.config_profiles
        assert "dev" in cfg.config_profiles

    def test_config_names_override(self, train_dag_path):
        """CLI --config overrides the DAG config: key."""
        cfg = load_workflow_config(train_dag_path, config_names=["global"])
        assert cfg.config_profiles == ["global"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_workflow_config(tmp_path / "nonexistent.yaml")

    def test_config_profile_merging(self, tmp_path):
        """Config profiles are merged: later overrides earlier."""
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "base.yaml").write_text("log_level: INFO\n")
        (cfg_dir / "override.yaml").write_text("log_level: DEBUG\n")

        dag_file = tmp_path / "train.yaml"
        dag_file.write_text(
            "workflow_name: test\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: test\nconfig_version: 1\n"
            "models: []\n"
        )

        cfg = load_workflow_config(dag_file, config_names=["base", "override"], config_dir=cfg_dir)
        assert cfg.log_level == "DEBUG"

    def test_dag_values_override_profiles(self, tmp_path):
        """DAG YAML values override merged profile values."""
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "global.yaml").write_text("log_level: DEBUG\n")

        dag_file = tmp_path / "train.yaml"
        dag_file.write_text(
            "workflow_name: test\nexecution_mode: sequential\n"
            "pipeline_type: training\nfeature_name: test\nconfig_version: 1\n"
            "log_level: WARNING\nmodels: []\n"
        )

        cfg = load_workflow_config(dag_file, config_names=["global"], config_dir=cfg_dir)
        assert cfg.log_level == "WARNING"
