"""Core framework abstractions."""

from mlplatform.core.context import ExecutionContext
from mlplatform.core.enums import ExecutionNature, ExecutionTarget, WorkloadType
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.steps import InferenceStep, Step, TrainStep
from mlplatform.core.trainer import BaseTrainer

__all__ = [
    "ExecutionContext",
    "WorkloadType",
    "ExecutionNature",
    "ExecutionTarget",
    "Step",
    "TrainStep",
    "InferenceStep",
    "BaseTrainer",
    "BasePredictor",
]
