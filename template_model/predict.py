"""Inference step: MyInference."""

from typing import Any

import pandas as pd

from mlplatform.core import InferenceStep
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.storage.base import Storage


class MyInference(InferenceStep, BasePredictor):
    """Load model and run predictions on inference data.

    Implements InferenceStep for local execution (run) and BasePredictor for
    Spark mapInPandas (load_model, predict_chunk).
    """

    def load_model(self, storage: Storage, path: str) -> Any:
        """Load model from storage. Used by Spark mapInPandas."""
        self._model = storage.load(path)
        return self._model

    def predict_chunk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a chunk. Used by Spark mapInPandas."""
        model = getattr(self, "_model", None)
        if model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")
        feature_cols = [
            c for c in data.columns if str(c) != "target" and not str(c).startswith("pred")
        ]
        X = data[feature_cols] if feature_cols else data
        predictions = model.predict(X)
        return data.assign(prediction=predictions)

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        """Local execution: load model and run predictions on inference data."""
        inference_data = kwargs.get("inference_data")
        if inference_data is None:
            raise ValueError("inference_data required")
        df = inference_data if isinstance(inference_data, pd.DataFrame) else pd.DataFrame(inference_data)
        model = self.load_artifact("model.pkl")
        feature_cols = [c for c in df.columns if str(c) != "target" and not str(c).startswith("pred")]
        X = df[feature_cols] if feature_cols else df
        predictions = model.predict(X)
        return predictions
