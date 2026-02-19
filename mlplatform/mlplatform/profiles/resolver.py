"""PrimitiveResolver - deterministic step + profile -> (runner, context) resolution."""

from __future__ import annotations

from typing import Any, Optional, Union

from mlplatform.core.context import ExecutionContext
from mlplatform.core.enums import ExecutionNature, WorkloadType
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.steps import Step
from mlplatform.core.trainer import BaseTrainer
from mlplatform.profiles.profile import Profile
from mlplatform.runners.base import JobRunner, ServiceRunner


class PrimitiveResolver:
    """Resolve a Step + Profile into (runner, ExecutionContext).

    Resolution rules:
    - Runner is chosen by step.execution_nature (JOB -> job_runner, SERVICE -> service_runner)
    - ExperimentTracker is injected only for TRAINING workloads
    - InvocationStrategy: step override or profile default (inference only)
    - trainer/predictor are injected based on workload type
    """

    def resolve(
        self,
        step: Step,
        profile: Profile,
        trainer: Optional[BaseTrainer] = None,
        predictor: Optional[BasePredictor] = None,
        runtime_config: Optional[dict[str, Any]] = None,
        environment_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Union[JobRunner, ServiceRunner], ExecutionContext]:
        invocation = None
        if step.workload_type == WorkloadType.INFERENCE:
            invocation = profile.default_invocation_strategy

        experiment_tracker = None
        if step.workload_type == WorkloadType.TRAINING:
            experiment_tracker = profile.experiment_tracker

        context = ExecutionContext(
            storage=profile.storage,
            artifact_store=profile.artifact_store,
            experiment_tracker=experiment_tracker,
            invocation_strategy=invocation,
            runtime_config=runtime_config or {},
            environment_metadata=environment_metadata or {},
            trainer=trainer,
            predictor=predictor,
        )

        if step.execution_nature == ExecutionNature.JOB:
            return profile.job_runner, context

        if step.execution_nature == ExecutionNature.SERVICE:
            if profile.service_runner is None:
                raise ValueError(
                    f"Profile '{profile.name}' has no service_runner, "
                    f"but step requires ExecutionNature.SERVICE"
                )
            return profile.service_runner, context

        raise ValueError(f"Unknown execution_nature: {step.execution_nature}")
