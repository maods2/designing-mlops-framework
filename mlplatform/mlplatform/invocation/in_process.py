"""In-process invocation: load model, read data via framework, run predict_chunk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.data_io import load_prediction_input, write_prediction_output
from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.config.schema import ModelConfig
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class InProcessInvocation(InvocationStrategy):
    """Load model in the current process, ingest data from YAML-configured
    source (CSV/Parquet/BigQuery), call predict_chunk, and write output.
    """

    def invoke(
        self,
        predictor: BasePredictor,
        context: ExecutionContext,
        model_cfg: ModelConfig,
    ) -> Any:
        context.log.info("Loading model for prediction: %s", context.model_name)
        predictor.load_model()
        context.log.info("Model loaded: %s", context.model_name)

        data = load_prediction_input(model_cfg)
        result = predictor.predict_chunk(data)
        write_prediction_output(result, model_cfg)
        context.log.info("In-process prediction complete: %d rows", len(result))
        return result
