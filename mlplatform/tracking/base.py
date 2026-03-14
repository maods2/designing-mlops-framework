"""Abstract Experiment Tracking interface with lifecycle hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking backends.

    Supports both singular (log_metric, log_param) and batch
    (log_metrics, log_params) logging methods.

    Lifecycle hooks (start_run/end_run) have no-op defaults so that
    simple trackers (NoneTracker, LocalJsonTracker) work without
    implementing them. Stateful trackers (VertexAI, MLflow) override
    these for scoped runs.
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

    def start_run(self, run_name: str | None = None) -> None:
        """Start an experiment run. No-op by default."""

    def end_run(self) -> None:
        """End the current experiment run. No-op by default."""

    def __enter__(self) -> "ExperimentTracker":
        self.start_run()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_run()
