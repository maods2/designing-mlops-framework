"""No-op experiment tracker."""

from __future__ import annotations

from typing import Any

from mlplatform.tracking.base import ExperimentTracker


class NoneTracker(ExperimentTracker):
    """No-op tracker that discards all logs."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: str, artifact: Any) -> None:
        pass
