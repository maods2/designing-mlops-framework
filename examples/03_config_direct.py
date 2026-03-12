"""Example: TrainingConfig and PredictionConfig from kwargs dict or keyword args.

Run: python examples/03_config_direct.py
"""

import _bootstrap  # noqa: F401

from mlplatform.config import PredictionConfig, TrainingConfig

# From keyword args
train = TrainingConfig(
    model_name="churn_model",
    feature="churn",
    version="1.0",
    module="churn.train",
    platform="VertexAI",
)
print(train.artifact_base_path, train.is_cloud_training)

# From kwargs dict
pred_kw = {
    "model_name": "churn_model",
    "feature": "churn",
    "version": "1.0",
    "module": "churn.predict",
    "platform": "VertexAI",
    "prediction_dataset_name": "my_project",
    "prediction_table_name": "customers",
    "prediction_output_dataset_table": "my_project.churn_predictions",
    "unique_identifier_column": "customer_id",
    "predicted_label_column_name": "will_churn",
    "predicted_probability_column_name": "churn_probability",
}
pred = PredictionConfig(pred_kw)
print(pred.input_source, pred.artifact_base_path)
