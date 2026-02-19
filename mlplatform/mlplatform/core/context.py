"""ExecutionContext - unified context passed to trainer/predictor code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlplatform.artifacts.base import ArtifactStore
    from mlplatform.storage.base import Storage
    from mlplatform.tracking.base import ExperimentTracker


@dataclass
class ExecutionContext:
    """Context injected into trainer/predictor instances.

    Provides typed fields for the pipeline identity (feature_name, model_name,
    version), convenience helpers that hide artifact path construction from DS
    code, and optional experiment tracking.
    """

    storage: Storage
    artifact_store: ArtifactStore
    experiment_tracker: Optional[ExperimentTracker]
    feature_name: str
    model_name: str
    version: str
    optional_configs: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("mlplatform"))
    _pipeline_type: str = ""

    @property
    def artifact_base_path(self) -> str:
        return f"{self.feature_name}/{self.model_name}/{self.version}"

    def save_artifact(self, name: str, obj: Any) -> None:
        """Save an artifact under the current model's versioned path."""
        self.storage.save(f"{self.artifact_base_path}/{name}", obj)

    def load_artifact(self, name: str) -> Any:
        """Load an artifact from the current model's versioned path."""
        return self.storage.load(f"{self.artifact_base_path}/{name}")

    def load_artifact_from(self, model_name: str, version: str, name: str) -> Any:
        """Load artifact from a different model/version (e.g., for ensemble)."""
        if version == "latest":
            meta = self.artifact_store.resolve_model(model_name, "latest")
            path = meta["path"]
        else:
            path = f"{self.feature_name}/{model_name}/{version}"
        return self.storage.load(f"{path}/{name}")

    def log_params(self, params: dict[str, Any]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(metrics)

    def register_model(self, metadata: dict[str, Any] | None = None) -> None:
        """Register current model version in the artifact store."""
        self.artifact_store.register_model(self.model_name, {
            "version": self.version,
            "feature_name": self.feature_name,
            "path": self.artifact_base_path,
            **(metadata or {}),
        })
