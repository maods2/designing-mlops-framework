"""Abstract InvocationStrategy interface for inference execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.core.predictor import BasePredictor


class InvocationStrategy(ABC):
    """Defines how prediction logic is invoked. Inference-only concern."""

    @abstractmethod
    def invoke(self, predictor: "BasePredictor", **kwargs: Any) -> Any:
        """Invoke the predictor using this strategy."""
        ...
