"""Local in-process runner implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.runners.base import JobRunner, ServiceRunner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class LocalJobRunner(JobRunner):
    """Execute steps in-process on the local machine."""

    def execute(self, step: "Step", context: "ExecutionContext") -> Any:
        return step.run(context)


class LocalServiceRunner(ServiceRunner):
    """Start a service step locally (e.g., REST endpoint for online inference dev)."""

    def start(self, step: "Step", context: "ExecutionContext") -> Any:
        return step.run(context)
