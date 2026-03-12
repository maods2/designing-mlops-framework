"""Unit tests for mlplatform.config.loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mlplatform.config.loader import (
    _deep_merge,
    _load_config_profiles,
    load_config_profiles,
)


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
        assert _deep_merge(base, {}) == base

    def test_empty_base(self):
        override = {"a": 1}
        assert _deep_merge({}, override) == override


class TestLoadConfigProfiles:
    def test_empty(self):
        assert _load_config_profiles([], None) == {}

    def test_config_dir_none_returns_empty(self):
        assert _load_config_profiles(["global", "dev"], None) == {}

    def test_load_config_profiles_with_yaml_files(self):
        """Public load_config_profiles with temp YAML files."""
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "global.yaml").write_text("model_name: m\nfeature: f\nversion: v1\n")
            (config_dir / "dev.yaml").write_text("version: dev\nbase_path: /out\n")
            result = load_config_profiles(["global", "dev"], config_dir)
            assert result["model_name"] == "m"
            assert result["feature"] == "f"
            assert result["version"] == "dev"  # dev overrides
            assert result["base_path"] == "/out"

    def test_load_config_profiles_accepts_str_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.yaml").write_text("x: 1\n")
            result = load_config_profiles(["a"], tmp)
            assert result == {"x": 1}

    def test_load_config_profiles_missing_file_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "exists.yaml").write_text("a: 1\n")
            result = load_config_profiles(["exists", "missing"], tmp)
            assert result == {"a": 1}  # missing.yaml skipped
