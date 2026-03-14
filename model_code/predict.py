"""Prediction: MyPredictor — config-driven, no DAG dependency.

Run locally:
    python -m model_code.predict

What you need to change:
  - FEATURE_COLUMNS in constants.py (must match training)
  - input_path / output_path in config (or use defaults)
"""

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import model_code.constants as cons


class MyPredictor(BasePredictor):
    """Load model and scaler, run predictions on data."""

    def load_model(self):
        """Load model and scaler from storage."""
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
    from mlplatform.config.models import PipelineConfig
    from mlplatform.runner import dev_predict

    config = PipelineConfig.from_dict({
        "model_name": "churn_model",
        "feature": "churn",
        "pipeline_type": "prediction",
        "module": "model_code.predict:MyPredictor",
        "config_list": ["global", "dev"],
        "config_dir": "model_code/config",
    })
    result = dev_predict(config)
    print(result)
