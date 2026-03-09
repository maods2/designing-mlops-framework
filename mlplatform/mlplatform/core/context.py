"""ExecutionContext - unified context passed to trainer/predictor code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.profiles.registry import Profile
    from mlplatform.storage.base import Storage

from mlplatform.core.artifact_registry import ArtifactRegistry
from mlplatform.tracking.base import ExperimentTracker
from mlplatform.tracking.none import NoneTracker


@dataclass
class ExecutionContext:
    """Context injected into trainer/predictor instances.

    Provides typed fields for the pipeline identity (feature_name, model_name,
    version), an ArtifactRegistry for persistence, and experiment tracking.

    The ``experiment_tracker`` always defaults to :class:`NoneTracker` so that
    ``log_params`` / ``log_metrics`` never silently fail. Override it with a
    real tracker via the ``from_profile`` factory or by passing one directly.
    """

    artifacts: ArtifactRegistry
    experiment_tracker: ExperimentTracker = field(default_factory=NoneTracker)
    feature_name: str = ""
    model_name: str = ""
    version: str = ""
    optional_configs: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("mlplatform"))
    _pipeline_type: str = ""
    commit_hash: str | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile: Profile,
        feature_name: str,
        model_name: str,
        version: str,
        base_path: str = "./artifacts",
        pipeline_type: str = "",
        log_level: str = "INFO",
        optional_configs: dict[str, Any] | None = None,
        commit_hash: str | None = None,
    ) -> ExecutionContext:
        """Build an ``ExecutionContext`` from a resolved :class:`Profile`.

        This is the **single, canonical** way to construct a context.  All
        orchestration code (runner, Spark entry point, Spark partition
        functions) should use this factory instead of duplicating the
        construction logic.
        """
        from mlplatform.utils.logging import get_logger

        storage = profile.storage_factory(base_path, profile.extra)
        tracker = profile.tracker_factory(base_path, profile.extra)
        registry = ArtifactRegistry(
            storage=storage,
            feature_name=feature_name,
            model_name=model_name,
            version=version,
        )
        return cls(
            artifacts=registry,
            experiment_tracker=tracker,
            feature_name=feature_name,
            model_name=model_name,
            version=version,
            optional_configs=optional_configs or {},
            log=get_logger(f"mlplatform.{model_name}", log_level),
            _pipeline_type=pipeline_type,
            commit_hash=commit_hash,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

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
        """Log hyperparameters via the experiment tracker."""
        self.experiment_tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics via the experiment tracker."""
        self.experiment_tracker.log_metrics(metrics)
