"""Backward-compatible shim — prediction schema has moved to mlplatform.core.prediction_schema."""

from mlplatform.core.prediction_schema import (  # noqa: F401
    PredictionInputSchema,
    SchemaValidationError,
    from_feature_columns,
    get_schema_from_predictor,
)

__all__ = [
    "PredictionInputSchema",
    "SchemaValidationError",
    "from_feature_columns",
    "get_schema_from_predictor",
]
