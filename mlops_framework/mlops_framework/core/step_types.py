"""Semantic step types for ML pipelines."""

from mlops_framework.core.step import BaseStep


class PreprocessStep(BaseStep):
    """
    Step for data preprocessing.

    Typically loads raw data, cleans/transforms it, and saves
    processed artifacts (e.g. train_data) for downstream steps.
    """

    input_schema = {"raw_data": "PATH"}
    output_schema = {"train_data": "DATASET"}


class TrainStep(BaseStep):
    """
    Step for model training.

    Typically loads preprocessed data, trains a model, saves the model
    artifact, and logs metrics. Experiment tracking is enabled only for TrainStep.
    """

    input_schema = {"train_data": "DATASET"}
    output_schema = {"model": "MODEL"}


class InferenceStep(BaseStep):
    """
    Step for inference/prediction.

    Typically loads a trained model, loads inference data, generates
    predictions, and saves the predictions artifact.
    """

    input_schema = {"model": "MODEL", "inference_data": "PATH"}
    output_schema = {"predictions": "DATASET"}


class DataDriftStep(BaseStep):
    """
    Step for data drift detection.

    Compares reference (training) distribution vs current (production) distribution.
    Outputs drift metrics (e.g. PSI, KS) and persists to tracking.
    """

    input_schema = {"reference_data": "DATASET", "current_data": "DATASET"}
    output_schema = {"drift_report": "REPORT"}


class ModelMonitorStep(BaseStep):
    """
    Step for model performance monitoring.

    Evaluates model on recent labeled data; tracks accuracy, prediction distribution.
    Outputs monitoring report and persists metrics to tracking.
    """

    input_schema = {"model": "MODEL", "validation_data": "DATASET"}
    output_schema = {"monitoring_report": "REPORT"}
