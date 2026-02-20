"""Prediction: MyPredictor - implements BasePredictor.

Run directly for local development/debugging:
    python example_model/predict.py
    python -m example_model.predict
"""

import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import example_model.constants as cons


class MyPredictor(BasePredictor):
    """Load model and scaler, run predictions on inference data."""

    def load_model(self) -> Any:
        """Load model and scaler from storage using context helpers."""
        self._model = self.context.load_artifact(cons.MODEL_ARTIFACT)
        self._scaler = self.context.load_artifact(cons.SCALER_ARTIFACT)
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


if __name__ == "__main__":
    from mlplatform.runner import run_workflow

    results = run_workflow(
        dag_path="template_prediction_dag.yaml",
        profile="local",
        version="dev",
    )
    for name, status in results.items():
        print(f"{name}: {status}")
