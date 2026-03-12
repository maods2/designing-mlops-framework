"""Example: train and predict with Artifact API.

Run: python examples/06_train_predict_workflow.py
"""

from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from mlplatform import Artifact
from mlplatform.config import TrainingConfig
from mlplatform.utils import HTMLReport

OUTPUT = str(Path(__file__).parent / "output" / "train_predict_demo")
CONFIG = TrainingConfig(
    model_name="sample_model",
    feature="demo",
    version="v1",
    base_path=OUTPUT,
    backend="local",
)


def train(config: TrainingConfig):
    artifact = Artifact(**config.to_artifact_kwargs())
    x = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
    y = x.pop("target")

    model = _SimpleModel()
    model.fit(x, y)
    preds = model.predict(x)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }

    artifact.save("model.pkl", model)
    artifact.save("metrics.json", metrics)  # sanitized automatically for .json

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(model.history["loss"])
    artifact.save("report/loss.png", fig)
    plt.close(fig)

    report = HTMLReport(title="Model Report", feature_name=config.feature)
    for k, v in metrics.items():
        report.add_metric(k, v)  # sanitized internally
    report.add_plot("loss", "report/loss.png")
    artifact.save("report.html", report.to_html())

    return artifact


def predict(config: TrainingConfig):
    artifact = Artifact(**config.to_artifact_kwargs())
    model = artifact.load("model.pkl")
    x = pd.DataFrame({"a": [2, 3], "b": [5, 6]})
    return model.predict(x)


class _SimpleModel:
    def fit(self, X, y):
        self.history = {"loss": [0.5, 0.3, 0.2]}
        return self

    def predict(self, X):
        return (X.iloc[:, 0] > 1.5).astype(int).values


if __name__ == "__main__":
    train(CONFIG)
    preds = predict(CONFIG)
