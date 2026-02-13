"""Vertex AI execution runner (stub for cloud pipelines)."""

from typing import Any, Dict, Optional, Type

from mlops_framework.backends.execution.base import BaseRunner
from mlops_framework.backends.storage.gcs import GCSStorage
from mlops_framework.backends.tracking.noop import NoOpTracker
from mlops_framework.backends.tracking.vertex import VertexTracker
from mlops_framework.core.context import ExecutionContext
from mlops_framework.core.step import BaseStep
from mlops_framework.core.step_types import TrainStep, DataDriftStep, ModelMonitorStep

MONITORING_STEP_TYPES = (TrainStep, DataDriftStep, ModelMonitorStep)


class VertexRunner(BaseRunner):
    """
    Stub for Vertex AI execution.
    
    Builds ExecutionContext with GCSStorage and VertexTracker.
    To implement: wire to Vertex Pipelines / Training jobs.
    """
    
    def __init__(
        self,
        bucket_name: str,
        experiment_name: str = "default",
        run_id: Optional[str] = None,
    ):
        """
        Initialize Vertex runner (stub).
        
        Args:
            bucket_name: GCS bucket for artifacts
            experiment_name: Vertex experiment name
            run_id: Optional run ID
        """
        self.bucket_name = bucket_name
        self.experiment_name = experiment_name
        self.run_id = run_id
    
    def run_step(
        self,
        step_class: Type[BaseStep],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run step with GCS storage. Tracking for TrainStep, DataDriftStep, ModelMonitorStep."""
        run_context = getattr(self, "run_context", None)
        use_tracking = issubclass(step_class, MONITORING_STEP_TYPES)
        tracker = (
            VertexTracker(
                experiment_name=run_context.experiment_name if run_context else self.experiment_name,
                run_name=self.run_id,
                project=getattr(run_context, "vertex_project", None) if run_context else None,
                location=getattr(run_context, "vertex_location", None) if run_context else None,
            )
            if use_tracking
            else NoOpTracker(run_id=self.run_id)
        )
        context = ExecutionContext(
            storage=GCSStorage(bucket_name=self.bucket_name),
            tracker=tracker,
            logger=lambda msg: print(f"[Vertex] {msg}"),
            config=config or {},
            run_context=run_context,
        )
        step = step_class(context)
        step.run()
