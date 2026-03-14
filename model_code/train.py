"""Training: MyTrainer — config-driven, no DAG dependency.

Run locally:
    python -m model_code.train

What you need to change:
  - FEATURE_COLUMNS in constants.py (your feature column names)
  - train_data_path in config/dev.yaml (path to your CSV)
  - Model and hyperparameters in the train() method below
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import model_code.constants as cons
from model_code.evaluate import evaluate


class MyTrainer(BaseTrainer):
    """Train a model from merged config template."""

    def train(self) -> None:
        # 1. Load data (CSV path from config, or train_data dict for tests)
        train_data = self.config.get("train_data")
        if train_data is not None:
            df = pd.concat(
                [train_data["X"], train_data["y"].rename("target")],
                axis=1,
            )
        else:
            data_path = self.config.get("train_data_path", "model_code/data/sample_train.csv")
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

        # 3. Train model
        hyperparams = self.config.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 4. Evaluate and log metrics
        val_df = pd.DataFrame(X_val, columns=cons.FEATURE_COLUMNS)
        val_df["target"] = y_val.values
        metrics = evaluate(model, scaler, val_df)
        self.log.info("Validation metrics: %s", metrics)
        self.tracker.log_metrics(metrics)
        self.tracker.log_params({"model_type": "LogisticRegression", "max_iter": max_iter})

        # 5. Save model and scaler
        self.artifacts.save(cons.MODEL_ARTIFACT, model)
        self.artifacts.save(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.config.models import PipelineConfig
    from mlplatform.runner import dev_train

    # Config from template — no DAG, no scattered args
    config = PipelineConfig.from_dict({
        "model_name": "churn_model",
        "feature": "churn",
        "pipeline_type": "training",
        "module": "model_code.train:MyTrainer",
        "config_list": ["global", "dev"],
        "config_dir": "model_code/config",
    })
    dev_train(config)
