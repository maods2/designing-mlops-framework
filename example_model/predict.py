"""Prediction: MyPredictor - implements BasePredictor."""

from typing import Any

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import example_model.constants as cons


class MyPredictor(BasePredictor):
    """Load model and scaler, run predictions on inference data.

    Implements BasePredictor for both local and Spark mapInPandas execution.
    Accesses self.context (ExecutionContext) for storage and runtime_config.
    """

    def load_model(self) -> Any:
        """Load model and scaler from storage using context."""
        ctx = self.context
        storage = ctx.storage
        runtime = ctx.runtime_config

        feature_name = runtime.get("feature_name", "default")
        model_name = runtime.get("model_name", "default")
        version = runtime.get("version", "dev")
        base_path = f"{feature_name}/{model_name}/{version}"

        self._model = storage.load(f"{base_path}/{cons.MODEL_ARTIFACT}")
        self._scaler = storage.load(f"{base_path}/{cons.SCALER_ARTIFACT}")
        return self._model

    def predict_chunk(self, data: Any) -> pd.DataFrame:
        """Run prediction on a chunk of data."""
        model = getattr(self, "_model", None)
        scaler = getattr(self, "_scaler", None)
        if model is None or scaler is None:
            raise RuntimeError("Model/scaler not loaded. Call load_model first.")

        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        feature_cols = [c for c in df.columns if c in cons.FEATURE_COLUMNS]
        if not feature_cols:
            feature_cols = [c for c in df.columns if str(c) != "target" and not str(c).startswith("pred")]
        X = df[feature_cols] if feature_cols else df
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return df.assign(prediction=predictions)
