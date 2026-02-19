"""In-process invocation strategy for local batch inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.core.predictor import BasePredictor


class InProcessInvocation(InvocationStrategy):
    """Calls predictor.predict_chunk() directly in the current process."""

    def invoke(self, predictor: "BasePredictor", **kwargs: Any) -> Any:
        data = kwargs.get("data")
        if data is None:
            raise ValueError("InProcessInvocation requires 'data' kwarg")
        return predictor.predict_chunk(data)
