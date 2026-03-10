"""Prediction: TrendPredictor - simple trend prediction example.

Run locally:
    python example_model_v2/predict.py

Uses ArtifactRegistry load and resolve_path for model artifact loading.
"""

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import example_model_v2.constants as cons


class TrendPredictor(BasePredictor):
    """Load model and scaler, run trend predictions on data."""

    def load_model(self):
        """Load model and scaler from storage via ArtifactRegistry."""
        self._model = self.artifacts.load(cons.MODEL_ARTIFACT)
        self._scaler = self.artifacts.load(cons.SCALER_ARTIFACT)
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

    result = dev_predict(dag_path="example_model_v2/pipeline/predict.yaml")
    print(result)
