"""Data drift detection step for part-failure pipeline."""

from custom.data_loader import create_synthetic_inference_data, load_raw_data
from custom.drift import compute_drift
from custom.feature_engineering import build_features
from mlops_framework.core.step_types import DataDriftStep


class PartFailureDataDrift(DataDriftStep):
    """
    Data drift step: compare reference (train) vs current (production) data.
    Uses custom.drift.compute_drift and persists metrics to tracking.
    """

    input_schema = {"reference_data": "DATASET", "current_data": "DATASET"}
    output_schema = {"drift_report": "REPORT"}

    def run(self) -> None:
        reference_artifact = self.context.config.get("reference_artifact", "train_data")
        current_data_path = self.context.config.get("current_data_path", "data/production_sample.csv")

        self.log(f"Loading reference data from artifact {reference_artifact}")
        train_data = self.load_artifact(reference_artifact)
        reference_df = train_data["X_train"]

        self.log(f"Loading current data from {current_data_path}")
        current_df = load_raw_data(current_data_path)
        if current_df is None:
            self.log("Current data not found. Using synthetic data for demo...")
            current_df = create_synthetic_inference_data(seed=123)
        else:
            current_df = build_features(current_df.dropna(), drop_target=False)

        report = compute_drift(reference_df, current_df)

        self.log_param("reference_artifact", reference_artifact)
        self.log_param("current_data_path", current_data_path)
        self.log_metric("psi_score", report["psi_score"])
        self.log_metric("max_drift", report["max_drift"])
        self.log_metric("n_features", report["n_features"])

        self.save_artifact("drift_report", report)
        self.log(f"Saved drift_report: psi={report['psi_score']}, max_drift={report['max_drift']}")


if __name__ == "__main__":
    from mlops_framework import run_local

    run_local("steps.data_drift.PartFailureDataDrift", step_id="data_drift")
