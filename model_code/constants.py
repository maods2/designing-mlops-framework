"""Model constants — artifact identity and feature definitions."""

MODEL_ARTIFACT = "model.pkl"
SCALER_ARTIFACT = "scaler.pkl"
FEATURE_COLUMNS = ["f0", "f1", "f2", "f3", "f4"]

# Artifact identity — defined once, used everywhere
ARTIFACT_IDENTITY = {
    "model_name": "churn_model",
    "feature": "churn",
}
