"""Integration tests for config profile merging."""

from __future__ import annotations

from pathlib import Path

from mlplatform.config import load_config_profiles
from mlplatform.config.loader import _load_config_profiles


class TestConfigFileMerging:
    def test_load_config_profiles_empty(self):
        result = _load_config_profiles([], None)
        assert result == {}

    def test_load_config_profiles_merges_example_configs(self):
        """Load global + dev from examples/config and verify merge."""
        config_dir = Path(__file__).parent.parent.parent / "examples" / "config"
        result = load_config_profiles(["global", "dev"], config_dir)
        assert result["model_name"] == "sample_model"
        assert result["feature"] == "demo"
        assert result["version"] == "dev"  # dev overrides global
        assert "dev_artifacts" in result["base_path"]
