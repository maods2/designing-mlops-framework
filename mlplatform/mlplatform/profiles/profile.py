"""Profile dataclass defining infrastructure defaults for an environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mlplatform.artifacts.base import ArtifactStore
from mlplatform.core.enums import ExecutionTarget
from mlplatform.etb.base import ExperimentTracker
from mlplatform.invocation.base import InvocationStrategy
from mlplatform.runners.base import JobRunner, ServiceRunner
from mlplatform.storage.base import Storage


@dataclass
class Profile:
    """Environment-oriented infrastructure configuration.

    Profiles define infrastructure defaults only. They do NOT encode
    training vs inference behavior or step intent.
    """

    name: str
    execution_target: ExecutionTarget
    job_runner: JobRunner
    service_runner: Optional[ServiceRunner]
    storage: Storage
    artifact_store: ArtifactStore
    experiment_tracker: Optional[ExperimentTracker]
    default_invocation_strategy: Optional[InvocationStrategy]
