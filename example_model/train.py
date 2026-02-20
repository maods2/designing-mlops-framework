"""Training: MyTrainer - implements BaseTrainer.

Data loading is the data scientist's responsibility.  This example shows
two paths:
  - **Local / CSV**: reads from a file (default ``example_model/data/sample_train.csv``).
  - **BigQuery**: renders a parameterised SQL template from ``sql/load_training_data.sql``
    using values supplied in ``optional_configs.bq_params``.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import example_model.constants as cons
from example_model.evaluate import evaluate
from example_model.utils import load_file, load_sql_template, render_sql


class MyTrainer(BaseTrainer):
    """Train LogisticRegression with StandardScaler, evaluate, and persist artifacts."""

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load training data - DS is responsible for this implementation.

        Checks ``optional_configs`` for a BigQuery param block first; falls
        back to a local CSV/Parquet file path.
        """
        ctx = self.context
        bq_params: dict | None = ctx.optional_configs.get("bq_params")

        if bq_params:
            sql_template = load_sql_template("load_training_data.sql")
            sql = render_sql(sql_template, bq_params)
            ctx.log.info("Loading training data from BigQuery:\n%s", sql)
            from example_model.utils import run_bq_query
            df = run_bq_query(sql)
        else:
            data_path = ctx.optional_configs.get(
                "train_data_path",
                str(Path(__file__).parent / "data" / "sample_train.csv"),
            )
            ctx.log.info("Loading training data from file: %s", data_path)
            df = load_file(data_path)

        X = df[cons.FEATURE_COLUMNS]
        y = df["target"]
        return X, y

    def train(self) -> None:
        ctx = self.context

        X, y = self._load_data()

        test_size = ctx.optional_configs.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        hyperparams = ctx.optional_configs.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)

        val_df = pd.DataFrame(X_val, columns=cons.FEATURE_COLUMNS)
        val_df["target"] = y_val.values
        metrics = evaluate(model, scaler, val_df)

        ctx.log.info("Validation metrics: %s", metrics)
        ctx.log_params({"model_type": "LogisticRegression", "max_iter": max_iter})
        ctx.log_metrics(metrics)

        ctx.save_artifact(cons.MODEL_ARTIFACT, model)
        ctx.save_artifact(cons.SCALER_ARTIFACT, scaler)


if __name__ == "__main__":
    from mlplatform.runner import dev_context

    ctx = dev_context("example_model/pipeline/dags/train.yaml")
    trainer = MyTrainer()
    trainer.context = ctx
    trainer.train()
