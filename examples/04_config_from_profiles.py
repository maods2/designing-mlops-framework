"""Example: load config from YAML profiles and build TrainingConfig.

Run: python examples/04_config_from_profiles.py

Shows load_config_profiles() — loads and deep-merges YAML files in order.
Later profiles override earlier ones. Use with TrainingConfig or PredictionConfig.
"""

from pathlib import Path

import _bootstrap  # noqa: F401

from mlplatform.config import TrainingConfig, load_config_profiles

CONFIG_DIR = Path(__file__).parent / "config"

# Load global + dev profiles (dev overrides global)
merged = load_config_profiles(["global", "dev"], CONFIG_DIR)
print("Merged config (global + dev):", merged)

cfg = TrainingConfig(merged)
print("TrainingConfig:", cfg.model_name, cfg.feature, cfg.version, cfg.base_path)
print("artifact_base_path:", cfg.artifact_base_path)

# Simulate prod: global + prod
merged_prod = load_config_profiles(["global", "prod"], CONFIG_DIR)
cfg_prod = TrainingConfig(merged_prod)
print("\nProd config:", cfg_prod.backend, cfg_prod.base_bucket, cfg_prod.version)
