"""Base class for execution runners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from mlops_framework.core.step import BaseStep


class BaseRunner(ABC):
    """Abstract base for local and cloud execution runners."""
    
    @abstractmethod
    def run_step(
        self,
        step_class: Type[BaseStep],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Execute a step with the appropriate context."""
        pass
