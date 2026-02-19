"""Abstract Runner interface for execution environments."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class Runner(ABC):
    """Abstract interface for execution environment runners."""

    @abstractmethod
    def run(self, step: "Step", context: "ExecutionContext", **kwargs: Any) -> Any:
        """Execute a step in this runner's environment."""
        ...
