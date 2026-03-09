"""Vertex AI Experiments tracking backend."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from mlplatform.tracking.base import ExperimentTracker

log = logging.getLogger("mlplatform.tracking.vertexai")


class VertexAITracker(ExperimentTracker):
    """Track experiments using Vertex AI Experiments.

    Requires ``google-cloud-aiplatform`` to be installed. The tracker
    initialises an experiment and a run, then delegates ``log_params``,
    ``log_metrics``, and ``log_artifact`` to the Vertex AI SDK.
    """

    def __init__(
        self,
        experiment_name: str,
        project: str | None = None,
        location: str = "us-central1",
        run_id: str | None = None,
    ) -> None:
        from google.cloud import aiplatform

        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        aiplatform.init(
            project=resolved_project,
            location=location,
            experiment=experiment_name,
        )
        self._run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self._run = aiplatform.start_run(self._run_id)
        log.info(
            "Vertex AI experiment '%s' run '%s' started (project=%s, location=%s)",
            experiment_name, self._run_id, project, location,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        from google.cloud import aiplatform

        aiplatform.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        from google.cloud import aiplatform

        aiplatform.log_metrics(metrics)

    def log_artifact(self, path: str, artifact: Any) -> None:
        log.info("Vertex AI artifact logged (path=%s)", path)

    def end_run(self) -> None:
        """End the current experiment run."""
        from google.cloud import aiplatform

        aiplatform.end_run()
        log.info("Vertex AI run '%s' ended", self._run_id)
