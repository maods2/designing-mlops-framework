"""Model monitoring step for part-failure pipeline."""

from custom.data_loader import create_synthetic_training_data, load_raw_data
from custom.feature_engineering import build_features
from custom.monitoring import compute_model_health
from mlops_framework.core.step_types import ModelMonitorStep


class PartFailureModelMonitor(ModelMonitorStep):
    """
    Model monitor step: evaluate model on recent labeled data.
    Uses custom.monitoring.compute_model_health and persists metrics to tracking.
    """

    input_schema = {"model": "MODEL", "validation_data": "DATASET"}
    output_schema = {"monitoring_report": "REPORT"}

    def run(self) -> None:
        validation_path = self.context.config.get("validation_path", "data/validation_latest.csv")

        self.log("Loading model artifact")
        model = self.load_artifact("model")

        self.log(f"Loading validation data from {validation_path}")
        df = load_raw_data(validation_path)
        if df is None:
            self.log("Validation data not found. Using synthetic data for demo...")
            X, y = create_synthetic_training_data(seed=999)
            X_val = X
            y_val = y
        else:
            df_clean = df.dropna()
            X_val = build_features(df_clean, drop_target=True)
            y_val = df_clean.iloc[:, -1]

        report = compute_model_health(model, X_val, y_val)

        self.log_param("validation_path", validation_path)
        for name, value in report.items():
            if isinstance(value, (int, float)):
                self.log_metric(name, float(value))

        self.save_artifact("monitoring_report", report)
        self.log(f"Saved monitoring_report: accuracy={report.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    from mlops_framework import run_local

    run_local("steps.model_monitor.PartFailureModelMonitor", step_id="model_monitor")
