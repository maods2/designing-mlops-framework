"""Integration tests for example_model training workflow."""

from __future__ import annotations

import pytest

from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.runner import run_workflow, _build_context


class TestRunTrainingWorkflow:
    def test_training_workflow_runs(self, train_dag_path, sample_train_data, artifacts_dir):
        """Training workflow completes and model artifact is saved."""
        pipeline = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, task_id="train_model", config_names=["global", "train-local"]
        )
        task_cfg = pipeline.tasks[0]
        ctx = _build_context(pipeline, task_cfg, "local", "test_v1", str(artifacts_dir))
        ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / pipeline.feature_name
            / task_cfg.model_name
            / "test_v1"
            / "model.pkl"
        )
        assert model_path.exists(), f"Model artifact not found at {model_path}"

    def test_pipeline_loads_and_trains(self, train_dag_path, sample_train_data, artifacts_dir):
        """New flat pipeline schema loads and trains."""
        pipeline = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, config_names=["global", "train-local"]
        )
        assert pipeline.pipeline_type == "training"
        train_tasks = [t for t in pipeline.tasks if t.module]
        assert len(train_tasks) >= 1

        task_cfg = train_tasks[0]
        ctx = _build_context(pipeline, task_cfg, "local", "new_dag_v1", str(artifacts_dir))
        ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / pipeline.feature_name
            / task_cfg.model_name
            / "new_dag_v1"
            / "model.pkl"
        )
        assert model_path.exists()


class TestDevPredict:
    def test_predict_with_data(
        self,
        train_dag_path,
        predict_dag_path,
        sample_train_data,
        sample_inference_df,
        artifacts_dir,
    ):
        """Train a model then run predict with explicit data."""
        import pandas as pd

        # Train first
        train_pipeline = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, task_id="train_model", config_names=["global", "train-local"]
        )
        train_task = train_pipeline.tasks[0]
        train_ctx = _build_context(
            train_pipeline, train_task, "local", "dev_test", str(artifacts_dir)
        )
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        # Predict
        pred_pipeline = ConfigLoaderFactory.load_pipeline_config(
            predict_dag_path, task_id="predict", config_names=["global", "predict-local"]
        )
        pred_task = pred_pipeline.tasks[0]
        pred_ctx = _build_context(
            pred_pipeline, pred_task, "local", "dev_test", str(artifacts_dir)
        )

        from example_model.predict import MyPredictor
        predictor = MyPredictor()
        predictor.context = pred_ctx
        predictor.load_model()
        result = predictor.predict(sample_inference_df)

        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert len(result) == len(sample_inference_df)

    def test_predict_method_name(self):
        """example_model predictor must expose predict() not predict_chunk()."""
        from example_model.predict import MyPredictor
        assert hasattr(MyPredictor, "predict")
        assert not hasattr(MyPredictor, "predict_chunk")
