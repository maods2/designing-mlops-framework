"""ExecutionContext - unified context passed to trainer/predictor code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
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

    def _resolve_artifact_path(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build the storage path for an artifact using path conventions."""
        m = model_name or self.model_name
        v = version or self.version
        return f"{self.feature_name}/{m}/{v}/{name}"

    def save_artifact(self, name: str, obj: Any) -> None:
        """Save an artifact under the current model's versioned path."""
        self.storage.save(f"{self.artifact_base_path}/{name}", obj)

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
        path = self._resolve_artifact_path(name, model_name=model_name, version=version)
        return self.storage.load(path)

    def log_params(self, params: dict[str, Any]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(metrics)
