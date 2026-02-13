"""Vertex AI tracking backend for cloud execution."""

from datetime import datetime
from typing import Any, Optional

from mlops_framework.backends.tracking.base import TrackingBackend

# Optional: google-cloud-aiplatform for real Vertex Experiments integration
# Install: pip install mlops-framework[vertex] or pip install google-cloud-aiplatform
_aiplatform = None
try:
    from google.cloud import aiplatform
    _aiplatform = aiplatform
except ImportError:
    pass


class VertexTracker(TrackingBackend):
    """
    Vertex AI Experiments tracking backend.

    When google-cloud-aiplatform is installed, logs metrics and params to
    Vertex AI Experiments. Otherwise falls back to no-op (interface ready).

    Requires: pip install google-cloud-aiplatform
    Optional extra: pip install mlops-framework[vertex]

    Before first use, call aiplatform.init(project=..., location=...) or set
    GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        run_name: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize Vertex tracker.

        Args:
            experiment_name: Vertex experiment name
            run_name: Optional run display name (auto-generated if None)
            project: GCP project (uses GOOGLE_CLOUD_PROJECT if not set)
            location: GCP region (uses GOOGLE_CLOUD_LOCATION or us-central1)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run_id = run_name or f"vertex_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._project = project
        self._location = location
        self._run = None
        if _aiplatform:
            try:
                import os
                proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
                loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
                if proj:
                    _aiplatform.init(project=proj, location=loc)
                self._run = _aiplatform.ExperimentRun(
                    run_name=self._run_id,
                    experiment=experiment_name,
                    project=proj,
                    location=loc,
                )
            except Exception:
                self._run = None

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to Vertex Experiments (or no-op if aiplatform unavailable)."""
        if self._run and _aiplatform:
            try:
                self._run.log_metrics({name: value})
            except Exception:
                pass

    def log_param(self, name: str, value: Any) -> None:
        """Log parameter to Vertex Experiments (or no-op if aiplatform unavailable)."""
        if self._run and _aiplatform:
            try:
                if not isinstance(value, (str, int, float, bool, type(None))):
                    value = str(value)
                self._run.log_params({name: value})
            except Exception:
                pass

    def get_run_id(self) -> str:
        """Return current run ID."""
        return self._run_id
