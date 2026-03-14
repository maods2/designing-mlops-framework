"""BasePredictor - serving abstraction for inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.core.artifact_registry import ArtifactRegistry
    from mlplatform.core.context import ExecutionContext
    from mlplatform.tracking.base import ExperimentTracker


class BasePredictor(ABC):
    """Base class for prediction. Same core used across batch-local, online-REST, batch-Spark.

    Implementations must define:
    - load_model(): load model artifacts (called before predict)
    - predict(data): run prediction on a chunk of data

    Use typed properties for better discoverability:

    - ``self.artifacts`` — load model artifacts (ArtifactRegistry)
    - ``self.tracker`` — log params and metrics (ExperimentTracker)
    - ``self.config`` — user_config dict
    - ``self.log`` — Logger
    """

    context: ExecutionContext

    @property
    def artifacts(self) -> ArtifactRegistry:
        """Load model artifacts. Use artifacts.load(name)."""
        return self.context.artifacts

    @property
    def tracker(self) -> ExperimentTracker:
        """Log params and metrics. Use tracker.log_params(dict), tracker.log_metrics(dict)."""
        return self.context.experiment_tracker

    @property
    def config(self) -> dict[str, Any]:
        """User config from YAML profiles (input_path, output_path, etc.)."""
        return self.context.optional_configs

    @property
    def log(self) -> Any:
        """Logger for this run."""
        return self.context.log

    def setup(self) -> None:
        """Called before :meth:`load_model`. Override for custom initialization."""

    @abstractmethod
    def load_model(self) -> Any:
        """Load model from storage."""
        ...

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Run prediction on a chunk of data. Returns predictions."""
        ...

    def teardown(self) -> None:
        """Called after prediction is complete (even on error). Override for cleanup."""
