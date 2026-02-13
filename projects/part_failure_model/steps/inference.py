"""Inference step for part-failure classification pipeline."""

import pandas as pd

from custom.data_loader import create_synthetic_inference_data, load_raw_data
from custom.feature_engineering import build_features
from mlops_framework.core.step_types import InferenceStep


class PartFailureInference(InferenceStep):
    """
    Inference step: load model, load inference data, build features, predict, save.
    Uses custom modules for business logic.
    """

    input_schema = {"model": "MODEL", "inference_data": "PATH"}
    output_schema = {"predictions": "DATASET"}

    def run(self) -> None:
        self.log("Loading model artifact")
        model = self.load_artifact("model")

        data_path = self.context.config.get("inference_path", "data/inference.csv")

        self.log(f"Loading inference data from {data_path}")
        data = load_raw_data(data_path)
        if data is None:
            self.log("Inference data not found. Creating synthetic data for demo...")
            data = create_synthetic_inference_data()

        X = build_features(data, drop_target=False)

        self.log("Generating predictions")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        preds = pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities,
        })
        self.save_artifact("predictions", preds)
        self.log(f"Saved predictions for {len(preds)} samples")


if __name__ == "__main__":
    from mlops_framework import run_local

    run_local("steps.inference.PartFailureInference")
