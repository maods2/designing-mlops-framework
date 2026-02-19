"""Training: MyTrainer - implements BaseTrainer.

Run directly for local development/debugging:
    python example_model/train.py
    python -m example_model.train
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import example_model.constants as cons


class MyTrainer(BaseTrainer):
    """Train LogisticRegression with StandardScaler, save model and scaler, log accuracy."""

    def _load_data(self) -> tuple:
        """Load training data. DS is responsible for this implementation.

        Override or adapt this to load from CSV, Parquet, BigQuery, GCS, etc.
        Returns (X, y) where X is features and y is target.
        """
        data_path = self.context.optional_configs.get(
            "train_data_path",
            str(Path(__file__).parent / "data" / "sample_train.csv"),
        )
        self.context.log.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]
        return X, y

    def train(self) -> None:
        ctx = self.context

        X, y = self._load_data()

        test_size = ctx.optional_configs.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        hyperparams = ctx.optional_configs.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)

        ctx.log.info("Accuracy: %.4f", accuracy)
        ctx.log_params({"model": "LogisticRegression", "max_iter": max_iter})
        ctx.log_metrics({"accuracy": float(accuracy)})

        ctx.save_artifact(cons.MODEL_ARTIFACT, model)
        ctx.save_artifact(cons.SCALER_ARTIFACT, scaler)
        ctx.register_model({"accuracy": float(accuracy)})


if __name__ == "__main__":
    from mlplatform.runner import dev_context

    ctx = dev_context("template_training_dag.yaml")
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()
