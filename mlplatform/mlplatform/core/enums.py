"""Core domain enums defining the four orthogonal execution axes."""

from __future__ import annotations

from enum import Enum


class WorkloadType(Enum):
    """What is being executed - the business domain of execution."""

    TRAINING = "training"
    INFERENCE = "inference"


class ExecutionNature(Enum):
    """How it executes - the lifecycle model."""

    JOB = "job"
    SERVICE = "service"


class ExecutionTarget(Enum):
    """Where it executes - the infrastructure class."""

    LOCAL = "local"
    CLOUD = "cloud"
    EMULATED_CLOUD = "emulated_cloud"
