"""Training: MyTrainer - implements BaseTrainer."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlplatform.core.trainer import BaseTrainer

import example_model.constants as cons


class MyTrainer(BaseTrainer):
    """Train LogisticRegression with StandardScaler, save model and scaler, log accuracy.

    Accesses self.context (ExecutionContext) for storage, experiment_tracker,
    and runtime_config.
    """

    def train(self) -> None:
        ctx = self.context
        storage = ctx.storage
        tracker = ctx.experiment_tracker
        runtime = ctx.runtime_config

        feature_name = runtime.get("feature_name", "default")
        model_name = runtime.get("model_name", "default")
        version = runtime.get("version", "dev")
        optional = runtime.get("optional_configs", {})
        base_artifact_path = f"{feature_name}/{model_name}/{version}"

        train_data = runtime.get("train_data")
        if train_data is None:
            raise ValueError("train_data required in runtime_config (pass via --train-data or inject)")

        X = train_data["X"]
        y = train_data["y"]
        test_size = optional.get("test_size", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        hyperparams = optional.get("hyperparameters", {})
        max_iter = hyperparams.get("max_iter", 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)

        if tracker:
            tracker.log_params({"model": "LogisticRegression", "max_iter": max_iter})
            tracker.log_metrics({"accuracy": float(accuracy)})

        storage.save(f"{base_artifact_path}/{cons.MODEL_ARTIFACT}", model)
        storage.save(f"{base_artifact_path}/{cons.SCALER_ARTIFACT}", scaler)

        artifact_store = ctx.artifact_store
        artifact_store.register_model(model_name, {
            "version": version,
            "feature_name": feature_name,
            "accuracy": float(accuracy),
            "path": base_artifact_path,
        })
