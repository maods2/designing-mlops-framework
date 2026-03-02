"""Training: MyTrainer - simple example for data scientists.

Run locally:
    python example_model/train.py

What you need to change:
  - FEATURE_COLUMNS in constants.py (your feature column names)
  - train_data_path in pipeline YAML (path to your CSV)
  - Model and hyperparameters in the train() method below
"""

import sys
from pathlib import Path

# Add repo root and mlplatform to path (needed for local runs)
_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import example_model.constants as cons
from example_model.evaluate import evaluate


class MyTrainer(BaseTrainer):
    """Train a model, evaluate it, and save artifacts."""

    def train(self) -> None:
        ctx = self.context

        # 1. Load data (CSV path from config or default)
        data_path = ctx.optional_configs.get(
            "train_data_path"
        )
        df = pd.read_csv(data_path)
        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]

        # 2. Split and scale
        test_size = ctx.optional_configs.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 3. Train model (change hyperparameters here if needed)
        hyperparams = ctx.optional_configs.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 4. Evaluate and log metrics
        val_df = pd.DataFrame(X_val, columns=cons.FEATURE_COLUMNS)
        val_df["target"] = y_val.values
        metrics = evaluate(model, scaler, val_df)
        ctx.log.info("Validation metrics: %s", metrics)
        ctx.log_metrics(metrics)
        ctx.log_params({"model_type": "LogisticRegression", "max_iter": max_iter})

        # 5. Save model and scaler
        ctx.save_artifact(cons.MODEL_ARTIFACT, model)
        ctx.save_artifact(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.runner import dev_train_context

    ctx = dev_train_context(
        "example_model/pipeline/train.yaml",
        task_id="train_model",
        config_names=["global", "train-local"],
    )
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()
