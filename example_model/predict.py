"""Inference step: MyInference."""

from typing import Any

import pandas as pd

from mlplatform.core import InferenceStep
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.storage.base import Storage

import example_model.constants as cons


class MyInference(InferenceStep, BasePredictor):
    """Load model and scaler, run predictions on inference data.

    Implements InferenceStep for local execution (run) and BasePredictor for
    Spark mapInPandas (load_model, predict_chunk). Applies scaler transform before prediction.
    """

    def load_model(self, storage: Storage, path: str) -> Any:
        """Load model and scaler from storage. Used by Spark mapInPandas."""
        self._model = storage.load(path)
        base = path.rsplit("/", 1)[0] if "/" in path else "."
        scaler_path = f"{base}/{cons.SCALER_ARTIFACT}"
        self._scaler = storage.load(scaler_path)
        return self._model

    def predict_chunk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a chunk. Used by Spark mapInPandas."""
        model = getattr(self, "_model", None)
        scaler = getattr(self, "_scaler", None)
        if model is None or scaler is None:
            raise RuntimeError("Model/scaler not loaded. Call load_model first.")
        feature_cols = [c for c in data.columns if c in cons.FEATURE_COLUMNS]
        if not feature_cols:
            feature_cols = [c for c in data.columns if str(c) != "target" and not str(c).startswith("pred")]
        X = data[feature_cols] if feature_cols else data
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return data.assign(prediction=predictions)

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        """Local execution: load model and scaler, run predictions on inference data."""
        inference_data = kwargs.get("inference_data")
        if inference_data is None:
            raise ValueError("inference_data required")
        df = inference_data if isinstance(inference_data, pd.DataFrame) else pd.DataFrame(inference_data)
        model = self.load_artifact(cons.MODEL_ARTIFACT)
        scaler = self.load_artifact(cons.SCALER_ARTIFACT)
        feature_cols = [c for c in df.columns if c in cons.FEATURE_COLUMNS]
        if not feature_cols:
            feature_cols = [c for c in df.columns if str(c) != "target" and not str(c).startswith("pred")]
        X = df[feature_cols] if feature_cols else df
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return predictions
