"""Abstract Experiment Tracking Backend interface."""

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking backends."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters or configuration."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics (e.g., accuracy, loss)."""
        ...

    @abstractmethod
    def log_artifact(self, path: str, artifact: Any) -> None:
        """Log an artifact (e.g., model file path)."""
        ...
