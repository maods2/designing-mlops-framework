"""Inference step: MyInference."""

from typing import Any

import pandas as pd

from mlplatform.core import InferenceStep
from mlplatform.core.context import ExecutionContext


class MyInference(InferenceStep):
    """Load model and run predictions on inference data."""

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        inference_data = kwargs.get("inference_data")
        if inference_data is None:
            raise ValueError("inference_data required")
        df = inference_data if isinstance(inference_data, pd.DataFrame) else pd.DataFrame(inference_data)
        model = self.load_artifact("model.pkl")
        feature_cols = [c for c in df.columns if str(c) != "target" and not str(c).startswith("pred")]
        X = df[feature_cols] if feature_cols else df
        predictions = model.predict(X)
        return predictions
