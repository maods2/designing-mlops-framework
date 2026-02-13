"""Train step for part-failure classification pipeline."""

from custom.evaluation import compute_metrics
from mlops_framework.core.step_types import TrainStep
from model import PartFailureModel


class PartFailureTrain(TrainStep):
    """
    Train step: load train_data, train model, evaluate, save model and metrics.
    Uses custom modules for evaluation logic.
    """

    input_schema = {"train_data": "DATASET"}
    output_schema = {"model": "MODEL"}

    def run(self) -> None:
        self.log("Loading train_data artifact")
        train_data = self.load_artifact("train_data")
        X_train = train_data["X_train"]
        y_train = train_data["y_train"]
        X_val = train_data["X_val"]
        y_val = train_data["y_val"]

        n_estimators = self.context.config.get("n_estimators", 100)
        max_depth = self.context.config.get("max_depth", 10)
        random_state = self.context.config.get("random_state", 42)

        self.log_param("n_estimators", n_estimators)
        self.log_param("max_depth", max_depth)
        self.log_param("random_state", random_state)

        self.log("Training model")
        model = PartFailureModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_pred_proba)

        for name, value in metrics.items():
            self.log_metric(name, value)
        self.log_metric("n_samples", len(X_train))

        self.save_artifact("model", model)
        self.log("Saved model artifact")


if __name__ == "__main__":
    from mlops_framework import run_local

    run_local("steps.train.PartFailureTrain")
