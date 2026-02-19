"""Training steps: MyPreprocess and MyTrain."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mlplatform.core import PreprocessStep, TrainStep
from mlplatform.core.context import ExecutionContext
from template_model.utils import load_csv, save_csv
import template_model.constants as cons


class MyPreprocess(PreprocessStep):
    """Load CSV and save train_data artifact for MyTrain."""

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        data_path = kwargs.get("data_path") or self._context.custom.get("data_path")
        if data_path is None:
            raise ValueError("data_path required (CSV path)")
        df = load_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("CSV must have 'target' column")
        X = df.drop(columns=["target"])
        y = df["target"]
        train_data = {"X": X, "y": y}
        self.save_artifact("train_data.pkl", train_data)
        return train_data


class MyTrain(TrainStep):
    """Train LogisticRegression, save model, log accuracy."""

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        train_data = kwargs.get("train_data")
        if train_data is None:
            train_data = self.load_artifact("train_data.pkl")

        X = train_data["X"]
        y = train_data["y"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        max_iter = self._context.custom.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.log_params({"model": "LogisticRegression", "max_iter": max_iter})
        self.log_metrics({"accuracy": float(accuracy)})
        self.save_artifact(cons.MODEL_ARTIFACT, model)
        return model
