"""Abstract Experiment Tracking Backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking backends.

    Supports both singular (log_metric, log_param) and batch
    (log_metrics, log_params) logging methods.
    """

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

    def log_metric(self, name: str, value: float) -> None:
        """Log a single metric."""
        self.log_metrics({name: value})

    def log_param(self, name: str, value: Any) -> None:
        """Log a single parameter."""
        self.log_params({name: value})
