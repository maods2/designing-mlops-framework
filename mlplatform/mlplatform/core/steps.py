"""Step abstractions - TrainStep, InferenceStep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mlplatform.core.enums import ExecutionNature, WorkloadType


class Step(ABC):
    """Abstract base for pipeline steps.

    Each step declares its workload_type and execution_nature.
    Steps do not resolve infrastructure -- that is handled by the Profile/Resolver.
    """

    workload_type: WorkloadType
    execution_nature: ExecutionNature

    @abstractmethod
    def run(self, context: Any) -> Any:
        """Execute the step with the given ExecutionContext."""
        ...


class TrainStep(Step):
    """Training step: always a Job, delegates to context.trainer.train()."""

    workload_type = WorkloadType.TRAINING
    execution_nature = ExecutionNature.JOB

    def run(self, context: Any) -> Any:
        if context.trainer is None:
            raise RuntimeError("TrainStep requires a trainer on the ExecutionContext")
        context.trainer.train()


class InferenceStep(Step):
    """Inference step: delegates to context.invocation_strategy.invoke(predictor).

    execution_nature is set per-instance (JOB for batch, SERVICE for online).
    """

    workload_type = WorkloadType.INFERENCE

    def __init__(self, execution_nature: ExecutionNature = ExecutionNature.JOB) -> None:
        self.execution_nature = execution_nature

    def run(self, context: Any) -> Any:
        if context.predictor is None:
            raise RuntimeError("InferenceStep requires a predictor on the ExecutionContext")
        if context.invocation_strategy is None:
            raise RuntimeError("InferenceStep requires an invocation_strategy on the ExecutionContext")
        return context.invocation_strategy.invoke(context.predictor)
