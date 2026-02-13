"""Runtime system for managing execution context."""

import os
from typing import Optional
from mlops_framework.config.loader import ConfigLoader
from mlops_framework.tracking.interface import TrackingBackend
from mlops_framework.tracking.local import LocalTrackingBackend
from mlops_framework.artifacts.manager import ArtifactManager


class Runtime:
    """
    Manages execution context, configuration, and tracking.
    
    Singleton pattern for global access. Initializes tracking backend
    and artifact manager based on configuration.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super(Runtime, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize runtime (only once)."""
        if self._initialized:
            return
        
        self.config_path = config_path
        self.config = ConfigLoader(config_path)
        self.tracking_backend: Optional[TrackingBackend] = None
        self.artifact_manager: Optional[ArtifactManager] = None
        self.run_id: Optional[str] = None
        
        self._initialize()
        Runtime._initialized = True
    
    def _initialize(self) -> None:
        """Initialize tracking backend and artifact manager."""
        runtime_mode = self.config.get_runtime_mode()
        
        # Only initialize tracking if mode requires it
        if runtime_mode in ["local_with_tracking", "cloud"]:
            self.tracking_backend = self._create_tracking_backend()
            self.run_id = self.tracking_backend.get_run_id()
        else:
            # For local mode without tracking, still generate a run ID
            from mlops_framework.tracking.local import LocalTrackingBackend
            temp_backend = LocalTrackingBackend()
            self.run_id = temp_backend.get_run_id()
        
        # Initialize artifact manager
        artifacts_config = self.config.get_artifacts_config()
        base_path = artifacts_config.get("base_path", "./artifacts")
        self.artifact_manager = ArtifactManager(
            base_path=base_path,
            run_id=self.run_id,
            tracking_backend=self.tracking_backend
        )
    
    def _create_tracking_backend(self) -> TrackingBackend:
        """Create the appropriate tracking backend based on config."""
        backend_name = self.config.get_tracking_backend()
        tracking_config = self.config.get_tracking_config(backend_name)
        
        # Get run name from config
        run_name = self.config.get_run_name()
        
        if backend_name == "mlflow":
            try:
                from mlops_framework.tracking.mlflow_backend import MLflowTrackingBackend
                return MLflowTrackingBackend(
                    tracking_uri=tracking_config.get("tracking_uri", "./mlruns"),
                    experiment_name=tracking_config.get("experiment_name", "default"),
                    run_id=run_name,
                    fallback_to_local=True
                )
            except ImportError:
                # Fallback to local if MLflow not available
                backend_name = "local"
        
        if backend_name == "local":
            return LocalTrackingBackend(
                base_path=tracking_config.get("base_path", "./runs"),
                run_id=run_name
            )
        
        raise ValueError(f"Unknown tracking backend: {backend_name}")
    
    def get_tracking_backend(self) -> Optional[TrackingBackend]:
        """Get the current tracking backend (may be None in local mode)."""
        return self.tracking_backend
    
    def get_artifact_manager(self) -> ArtifactManager:
        """Get the artifact manager."""
        return self.artifact_manager
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id
    
    def get_config(self) -> ConfigLoader:
        """Get the configuration loader."""
        return self.config
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
        cls._initialized = False
