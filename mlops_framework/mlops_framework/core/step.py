"""Base step abstraction for ML pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mlops_framework.core.context import ExecutionContext


class BaseStep(ABC):
    """
    Base class for all pipeline steps.

    Data scientists inherit from PreprocessStep, TrainStep, or InferenceStep.
    They NEVER import GCS or Vertex — all infra is injected via context.

    FBLearner-inspired: input_schema and output_schema declare artifacts for DAG validation / UI.
    """

    # FBLearner-inspired: artifact name -> type hint (PATH, DATASET, MODEL, etc.)
    input_schema: Dict[str, str] = {}
    output_schema: Dict[str, str] = {}
    
    def __init__(self, context: ExecutionContext):
        """Initialize the step with execution context."""
        self.context = context
    
    @abstractmethod
    def run(self) -> None:
        """Execute the step. Must be implemented by subclasses."""
        pass
    
    # ---------- Artifact API ----------
    
    def save_artifact(self, name: str, obj: Any) -> None:
        """Save an artifact via the storage backend."""
        self.context.storage.save(name, obj)
    
    def load_artifact(self, name: str) -> Any:
        """Load an artifact from the storage backend."""
        return self.context.storage.load(name)
    
    # ---------- Metrics API ----------
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric via the tracking backend."""
        self.context.tracker.log_metric(name, value)
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter via the tracking backend."""
        self.context.tracker.log_param(name, value)
    
    # ---------- Logging ----------
    
    def log(self, message: str) -> None:
        """Log a message via the context logger."""
        self.context.logger(message)
