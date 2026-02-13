"""Configuration loader for YAML-based configuration."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Loads and validates YAML configuration files.
    
    Provides defaults for missing values and validates required fields.
    """
    
    DEFAULT_CONFIG = {
        "runtime": {
            "mode": "local",
            "tracking_backend": "local",
            "run_name": None,
        },
        "tracking": {
            "local": {
                "base_path": "./runs"
            },
            "mlflow": {
                "tracking_uri": "./mlruns",
                "experiment_name": "default"
            }
        },
        "artifacts": {
            "base_path": "./artifacts"
        }
    }
    
    VALID_RUNTIME_MODES = ["local", "local_with_tracking", "cloud"]
    VALID_TRACKING_BACKENDS = ["local", "mlflow"]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to YAML config file. If None, uses defaults only.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        config = self.DEFAULT_CONFIG.copy()
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                config = self._deep_merge(config, file_config)
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        runtime_mode = self.config.get("runtime", {}).get("mode", "local")
        if runtime_mode not in self.VALID_RUNTIME_MODES:
            raise ValueError(
                f"Invalid runtime mode: {runtime_mode}. "
                f"Must be one of: {self.VALID_RUNTIME_MODES}"
            )
        
        tracking_backend = self.config.get("runtime", {}).get("tracking_backend", "local")
        if tracking_backend not in self.VALID_TRACKING_BACKENDS:
            raise ValueError(
                f"Invalid tracking backend: {tracking_backend}. "
                f"Must be one of: {self.VALID_TRACKING_BACKENDS}"
            )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot-notation path.
        
        Args:
            key_path: Dot-separated path (e.g., "runtime.mode")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_runtime_mode(self) -> str:
        """Get the runtime mode."""
        return self.get("runtime.mode", "local")
    
    def get_tracking_backend(self) -> str:
        """Get the tracking backend name."""
        return self.get("runtime.tracking_backend", "local")
    
    def get_tracking_config(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific tracking backend."""
        if backend is None:
            backend = self.get_tracking_backend()
        return self.get(f"tracking.{backend}", {})
    
    def get_artifacts_config(self) -> Dict[str, Any]:
        """Get artifacts configuration."""
        return self.get("artifacts", {})
    
    def get_run_name(self) -> Optional[str]:
        """Get the run name (may be None for auto-generation)."""
        return self.get("runtime.run_name")
