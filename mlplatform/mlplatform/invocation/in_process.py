"""In-process invocation: load model and run predict_chunk directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class InProcessInvocation(InvocationStrategy):
    """Load model in the current process and call predict_chunk on input data.

    This is the default local invocation strategy. The predictor is responsible
    for loading its own input data via its internal ``_load_input_data`` method
    (if defined), or the caller supplies data externally.
    """

    def invoke(self, predictor: BasePredictor, context: ExecutionContext) -> Any:
        context.log.info("Loading model for prediction: %s", context.model_name)
        predictor.load_model()
        context.log.info("Model loaded: %s", context.model_name)

        load_input = getattr(predictor, "_load_input_data", None)
        if callable(load_input):
            data = load_input()
            result = predictor.predict_chunk(data)
            context.log.info("In-process prediction complete: %d rows", len(result))
            return result

        context.log.info("Model loaded and ready for predict_chunk calls")
        return None
