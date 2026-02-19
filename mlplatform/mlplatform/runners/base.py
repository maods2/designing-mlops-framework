"""Abstract Runner interfaces for execution environments."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class JobRunner(ABC):
    """Execute finite-lifecycle steps (training, batch inference)."""

    @abstractmethod
    def execute(self, step: "Step", context: "ExecutionContext") -> Any:
        """Execute a step as a job."""
        ...


class ServiceRunner(ABC):
    """Start long-lived service steps (online inference)."""

    @abstractmethod
    def start(self, step: "Step", context: "ExecutionContext") -> Any:
        """Start a step as a long-running service."""
        ...
