"""Prediction: MyPredictor - simple example for data scientists.

Run locally:
    python example_model/predict.py

What you need to change:
  - FEATURE_COLUMNS in constants.py (must match training)
  - input_path / output_path in config (or use defaults)
"""

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import example_model.constants as cons


class MyPredictor(BasePredictor):
    """Load model and scaler, run predictions on data."""

    def load_model(self):
        """Load model and scaler from storage."""
        self._model = self.context.load_artifact(cons.MODEL_ARTIFACT)
        self._scaler = self.context.load_artifact(cons.SCALER_ARTIFACT)
        return self._model

    def predict(self, data):
        """Run prediction. Returns input DataFrame with a 'prediction' column added."""

        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        X = df[cons.FEATURE_COLUMNS]
        X_scaled = self._scaler.transform(X)
        predictions = self._model.predict(X_scaled)
        return df.assign(prediction=predictions)


if __name__ == "__main__":
    from mlplatform.runner import dev_predict

    result = dev_predict(dag_path="example_model/pipeline/predict.yaml")
    print(result)
