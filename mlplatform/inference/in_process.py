"""In-process inference: load model, read data via framework, run predict."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.core.prediction_schema import get_schema_from_predictor
from mlplatform.data.io import load_prediction_input, write_prediction_output
from mlplatform.inference.base import InferenceStrategy

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class InProcessInference(InferenceStrategy):
    """Load model in the current process, ingest data from configured
    source (CSV/Parquet/BigQuery), call predict, and write output.
    """

    def invoke(
        self,
        predictor: BasePredictor,
        context: ExecutionContext,
        model_cfg: Any,
    ) -> Any:
        context.log.info("Loading model for prediction: %s", context.model_name)
        predictor.load_model()
        context.log.info("Model loaded: %s", context.model_name)

        data = load_prediction_input(model_cfg)
        schema = get_schema_from_predictor(predictor)
        if schema:
            schema.validate(data)
        result = predictor.predict(data)
        write_prediction_output(result, model_cfg)
        context.log.info("In-process prediction complete: %d rows", len(result))
        return result
