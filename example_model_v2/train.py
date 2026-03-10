"""Training: TrendTrainer - simple trend prediction example.

Run locally:
    python example_model_v2/train.py

Uses ArtifactRegistry save/load/resolve_path for model and scaler persistence.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import example_model_v2.constants as cons
from example_model_v2.evaluate import evaluate


class TrendTrainer(BaseTrainer):
    """Train a trend regression model, evaluate it, and save artifacts."""

    def train(self) -> None:
        # 1. Load data (CSV path from config, or train_data dict for tests)
        train_data = self.config.get("train_data")
        if train_data is not None:
            df = pd.concat(
                [train_data["X"], train_data["y"].rename("target")],
                axis=1,
            )
        else:
            data_path = self.config.get(
                "train_data_path", "example_model_v2/data/sample_trend.csv"
            )
            df = pd.read_csv(data_path)
        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]

        # 2. Split and scale
        test_size = self.config.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 3. Train model (LinearRegression for trend)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # 4. Evaluate and log metrics
        val_df = pd.DataFrame(X_val, columns=cons.FEATURE_COLUMNS)
        val_df["target"] = y_val.values
        metrics = evaluate(model, scaler, val_df)
        self.log.info("Validation metrics: %s", metrics)
        self.tracker.log_metrics(metrics)
        self.tracker.log_params({"model_type": "LinearRegression"})

        # 5. Save model and scaler using ArtifactRegistry
        self.artifacts.save(cons.MODEL_ARTIFACT, model)
        self.artifacts.save(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.runner import dev_train

    dev_train("example_model_v2/pipeline/train.yaml")
