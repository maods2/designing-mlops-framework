"""ExecutionContext - unified context passed to trainer/predictor code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlplatform.storage.base import Storage
    from mlplatform.tracking.base import ExperimentTracker

from mlplatform.core.artifact_registry import ArtifactRegistry


@dataclass
class ExecutionContext:
    """Context injected into trainer/predictor instances.

    Provides typed fields for the pipeline identity (feature_name, model_name,
    version), an ArtifactRegistry for persistence, and optional experiment
    tracking.
    """

    artifacts: ArtifactRegistry
    experiment_tracker: Optional[ExperimentTracker]
    feature_name: str
    model_name: str
    version: str
    optional_configs: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("mlplatform"))
    _pipeline_type: str = ""

    @property
    def storage(self) -> Storage:
        """Direct access to the underlying Storage backend."""
        return self.artifacts.storage

    def save_artifact(self, name: str, obj: Any) -> None:
        """Save an artifact under the current model's versioned path."""
        self.artifacts.save(name, obj)

    def load_artifact(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load an artifact. Defaults to current model/version.

        Override model_name/version for cross-model loading (e.g., ensembles).
        """
        return self.artifacts.load(name, model_name=model_name, version=version)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(metrics)
