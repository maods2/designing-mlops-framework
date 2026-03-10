"""Abstract InferenceStrategy interface for prediction execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.config.schema import ModelConfig
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class InferenceStrategy(ABC):
    """Defines how a predictor is invoked at runtime.

    Decouples the prediction contract (load_model + predict) from the
    transport and distribution mechanism (in-process, Spark, REST, etc.).
    """

    @abstractmethod
    def invoke(
        self,
        predictor: BasePredictor,
        context: ExecutionContext,
        model_cfg: ModelConfig,
    ) -> Any:
        """Execute prediction using the given predictor, context, and model config."""
        ...
