"""BaseTrainer - training abstraction contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.core.artifact_registry import ArtifactRegistry
    from mlplatform.core.context import ExecutionContext
    from mlplatform.tracking.base import ExperimentTracker


class BaseTrainer(ABC):
    """Base class for model training. Implement train() with your training logic.

    The trainer receives an ExecutionContext (set as self.context) before train()
    is called. Use typed properties for better discoverability:

    - ``self.artifacts`` — save/load model artifacts (ArtifactRegistry)
    - ``self.tracker`` — log params and metrics (ExperimentTracker)
    - ``self.config`` — user_config dict (merged from YAML profiles)
    - ``self.log`` — Logger
    """

    context: ExecutionContext

    @property
    def artifacts(self) -> ArtifactRegistry:
        """Save/load model artifacts. Use artifacts.save(name, obj) and artifacts.load(name)."""
        return self.context.artifacts

    @property
    def tracker(self) -> ExperimentTracker:
        """Log params and metrics. Use tracker.log_params(dict), tracker.log_metrics(dict)."""
        return self.context.experiment_tracker

    @property
    def config(self) -> dict[str, Any]:
        """User config from YAML profiles (train_data_path, hyperparameters, etc.)."""
        return self.context.optional_configs

    @property
    def log(self) -> Any:
        """Logger for this run."""
        return self.context.log

    def setup(self) -> None:
        """Called before :meth:`train`. Override for custom initialization."""

    @abstractmethod
    def train(self) -> None:
        """Execute the training workflow."""
        ...

    def teardown(self) -> None:
        """Called after :meth:`train` (even if train raised). Override for cleanup."""
