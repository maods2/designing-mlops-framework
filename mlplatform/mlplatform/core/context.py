"""ExecutionContext - unified context passed to trainer/predictor code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlplatform.storage.base import Storage
    from mlplatform.tracking.base import ExperimentTracker

from mlplatform.core.artifact_registry import ArtifactRegistryProtocol


def _get_by_dotted_key(d: dict[str, Any], key: str, default: Any) -> Any:
    """Look up a value by key, supporting dot-notation for nested keys."""
    parts = key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            return default
        if part not in current:
            return default
        current = current[part]
    return current


class ConfigView:
    """Dict-like view over optional_configs with dot-notation support for nested keys."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key. Supports dot-notation for nested keys (e.g. 'hyperparameters.max_iter')."""
        return _get_by_dotted_key(self._data, key, default)

    def __getitem__(self, key: str) -> Any:
        """Direct access by key. Supports dot-notation. Raises KeyError if missing."""
        value = _get_by_dotted_key(self._data, key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """Check if key exists. Supports dot-notation."""
        return _get_by_dotted_key(self._data, key, _MISSING) is not _MISSING


_MISSING = object()


@dataclass
class ExecutionContext:
    """Context injected into trainer/predictor instances.

    Provides typed fields for the pipeline identity (feature_name, model_name,
    version), an ArtifactRegistry for persistence, and optional experiment
    tracking.
    """

    artifacts: ArtifactRegistryProtocol
    experiment_tracker: Optional[ExperimentTracker]
    feature_name: str
    model_name: str
    version: str
    optional_configs: dict[str, Any] = field(default_factory=dict)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("mlplatform"))
    _pipeline_type: str = ""
    commit_hash: str | None = None

    @property
    def config(self) -> ConfigView:
        """Flat config view with dot-notation support. Use ctx.config.get('key') or ctx.config['key']."""
        return ConfigView(self.optional_configs)

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
