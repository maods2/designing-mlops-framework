"""Example: Artifact API — save and load model artifacts.

Run: python examples/05_artifacts_standalone.py

Uses Artifact() with explicit params. For config-driven usage, see
examples/06_train_predict_workflow.py with TrainingConfig.
"""

from pathlib import Path

import _bootstrap  # noqa: F401

from mlplatform import Artifact

OUTPUT = str(Path(__file__).parent / "output" / "artifacts_demo")

artifact = Artifact(
    model_name="sample_model",
    feature="demo",
    version="v1",
    base_path=OUTPUT,
    backend="local",
)

model_data = {"weights": [1.0, 2.0], "bias": 0.5}
artifact.save("model.pkl", model_data)
artifact.save("metrics.json", {"accuracy": 0.95})  # dict sanitized automatically

loaded = artifact.load("model.pkl")
metrics = artifact.load("metrics.json")
assert loaded == model_data
assert metrics["accuracy"] == 0.95
