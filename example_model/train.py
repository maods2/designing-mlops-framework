"""Training step: MyTrain."""

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mlplatform.core import TrainStep
from mlplatform.core.context import ExecutionContext
import example_model.constants as cons


class MyTrain(TrainStep):
    """Train LogisticRegression with StandardScaler, save model and scaler, log accuracy."""

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        train_data = kwargs.get("train_data")
        if train_data is None:
            raise ValueError("train_data required (pass via --train-data or step_kwargs)")

        X = train_data["X"]
        y = train_data["y"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        max_iter = context.custom.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)

        self.log_params({"model": "LogisticRegression", "max_iter": max_iter})
        self.log_metrics({"accuracy": float(accuracy)})
        self.save_artifact(cons.MODEL_ARTIFACT, model)
        self.save_artifact(cons.SCALER_ARTIFACT, scaler)
        return model
