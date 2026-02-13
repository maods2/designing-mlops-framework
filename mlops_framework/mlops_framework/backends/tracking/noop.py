"""No-op tracking backend (metrics/params discarded)."""

from datetime import datetime
from typing import Any, Optional

from mlops_framework.backends.tracking.base import TrackingBackend


class NoOpTracker(TrackingBackend):
    """
    No-op tracker for when experiment tracking is disabled.
    
    All methods are no-ops; metrics and params are discarded.
    Used locally by default (--tracking flag to enable LocalTracker).
    """

    def __init__(self, run_id: Optional[str] = None):
        self._run_id = run_id or f"noop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass

    def log_param(self, name: str, value: Any) -> None:
        pass

    def get_run_id(self) -> str:
        return self._run_id
