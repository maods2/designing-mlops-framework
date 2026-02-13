"""Local execution runner for step-based pipelines."""

import logging
from typing import Any, Dict, Literal, Optional, Type

from mlops_framework.backends.execution.base import BaseRunner
from mlops_framework.backends.storage.local import LocalStorage
from mlops_framework.backends.tracking.local import LocalTracker
from mlops_framework.backends.tracking.noop import NoOpTracker
from mlops_framework.backends.tracking.vertex import VertexTracker
from mlops_framework.core.context import ExecutionContext
from mlops_framework.core.run_context import RunContext
from mlops_framework.core.step import BaseStep
from mlops_framework.core.step_types import TrainStep, DataDriftStep, ModelMonitorStep

MONITORING_STEP_TYPES = (TrainStep, DataDriftStep, ModelMonitorStep)


def _resolve_tracker(
    tracking: bool,
    tracking_backend: Literal["noop", "local", "vertex"],
    runs_path: str,
    run_id: Optional[str],
    run_context: Optional[RunContext],
) -> Any:
    """Resolve TrackingBackend: noop, local, or vertex."""
    if tracking_backend == "vertex" and tracking:
        return VertexTracker(
            experiment_name=run_context.experiment_name if run_context else "default",
            run_name=run_id,
            project=run_context.vertex_project if run_context else None,
            location=run_context.vertex_location if run_context else None,
        )
    if tracking_backend == "local" or tracking:
        return LocalTracker(base_path=runs_path, run_id=run_id)
    return NoOpTracker(run_id=run_id)


class LocalRunner(BaseRunner):
    """
    Runs steps locally with LocalStorage and LocalTracker, VertexTracker, or NoOpTracker.
    
    No cloud access required for local/noop. Vertex backend requires GCP creds.
    Tracking off by default; set tracking=True or tracking_backend="local"/"vertex".
    """
    
    def __init__(
        self,
        artifacts_path: str = "./artifacts",
        runs_path: str = "./runs",
        run_id: Optional[str] = None,
        tracking: bool = False,
        tracking_backend: Literal["noop", "local", "vertex"] = "noop",
        run_context: Optional[RunContext] = None,
    ):
        """
        Initialize local runner.
        
        Args:
            artifacts_path: Base path for artifact storage
            runs_path: Base path for tracking runs (used when tracking_backend=local)
            run_id: Optional run ID (auto-generated if None)
            tracking: If True enable tracking (uses tracking_backend)
            tracking_backend: noop, local, or vertex
            run_context: Optional RunContext; if set, overrides paths and tracking
        """
        self.run_context = run_context
        if run_context:
            self.artifacts_path = run_context.artifacts_path or run_context.base_path
            self.runs_path = run_context.base_path.replace("artifacts", "runs", 1) if "artifacts" in run_context.base_path else f"{run_context.base_path}/runs"
            self.run_id = run_context.run_id
            self.tracking = run_context.tracking_enabled
            self.tracking_backend = run_context.tracking_backend
        else:
            self.artifacts_path = artifacts_path
            self.runs_path = runs_path
            self.run_id = run_id
            self.tracking = tracking
            self.tracking_backend = tracking_backend
    
    def run_step(
        self,
        step_class: Type[BaseStep],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run a single step locally.
        
        Args:
            step_class: Step class (e.g. PartFailurePreprocess)
            config: Optional configuration dict for the step
        """
        # Tracking for TrainStep, DataDriftStep, ModelMonitorStep; framework hides complexity
        use_tracking = issubclass(step_class, MONITORING_STEP_TYPES) and (
            self.tracking or self.tracking_backend in ("local", "vertex")
        )
        tracker = _resolve_tracker(
            use_tracking,
            self.tracking_backend if use_tracking else "noop",
            self.runs_path,
            self.run_id,
            self.run_context,
        )
        context = ExecutionContext(
            storage=LocalStorage(base_path=self.artifacts_path),
            tracker=tracker,
            logger=logging.getLogger("mlops").info,
            config=config or {},
            run_context=self.run_context,
        )
        step = step_class(context)
        step.run()
