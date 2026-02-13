"""Abstract base class for tracking backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TrackingBackend(ABC):
    """
    Abstract base class for experiment tracking backends.
    
    All tracking backends must implement this interface to provide
    consistent logging capabilities across different tracking systems.
    """
    
    @abstractmethod
    def log_param(self, name: str, value: Any) -> None:
        """
        Log a parameter (hyperparameter, configuration value, etc.).
        
        Args:
            name: Parameter name
            value: Parameter value (must be JSON-serializable)
        """
        pass
    
    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (for time-series metrics)
        """
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file) to the tracking system.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the tracking system
                          (if None, uses the filename from local_path)
        """
        pass
    
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a trained model.
        
        Args:
            model: The model object to log
            artifact_path: Path where the model should be stored
            metadata: Optional metadata dictionary about the model
        """
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tag name -> tag value
        """
        pass
    
    @abstractmethod
    def get_run_id(self) -> str:
        """
        Get the current run ID.
        
        Returns:
            Unique identifier for the current run
        """
        pass
