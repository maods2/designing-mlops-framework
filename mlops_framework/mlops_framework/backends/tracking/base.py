"""Abstract base class for tracking backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class TrackingBackend(ABC):
    """
    Minimal tracking interface for step-based flow.
    
    Supports log_metric, log_param, and get_run_id.
    Implementations: LocalTracker, VertexTracker.
    """
    
    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        pass
    
    @abstractmethod
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter (hyperparameter, config value, etc.)."""
        pass
    
    @abstractmethod
    def get_run_id(self) -> str:
        """Get the current run ID."""
        pass
