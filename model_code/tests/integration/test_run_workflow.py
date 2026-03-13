"""Integration tests for example_model training workflow."""

from __future__ import annotations

import pytest

from mlplatform.runner import run_workflow, _build_context
from mlplatform.config.loader import load_workflow_config


class TestRunTrainingWorkflow:
    def test_training_workflow_runs(self, legacy_train_dag_path, sample_train_data, artifacts_dir):
        """Training workflow completes and model artifact is saved."""
        workflow = load_workflow_config(legacy_train_dag_path)
        model_cfg = workflow.models[0]
        ctx = _build_context(workflow, model_cfg, "local", "test_v1", str(artifacts_dir))
        ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / workflow.feature_name
            / model_cfg.model_name
            / "test_v1"
            / "model.pkl"
        )
        assert model_path.exists(), f"Model artifact not found at {model_path}"

    def test_new_dag_format_loads_and_trains(self, train_dag_path, sample_train_data, artifacts_dir):
        """New DAG format with resources.jobs.deployment.tasks loads and trains."""
        workflow = load_workflow_config(train_dag_path)
        assert workflow.pipeline_type == "training"
        assert len(workflow.models) >= 1

        model_cfg = workflow.models[0]
        ctx = _build_context(workflow, model_cfg, "local", "new_dag_v1", str(artifacts_dir))
        ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / workflow.feature_name
            / model_cfg.model_name
            / "new_dag_v1"
            / "model.pkl"
        )
        assert model_path.exists()


class TestDevPredict:
    def test_predict_with_data(
        self,
        legacy_train_dag_path,
        legacy_predict_dag_path,
        sample_train_data,
        sample_inference_df,
        artifacts_dir,
    ):
        """Train a model then run predict with explicit data."""
        import pandas as pd

        # Train first
        train_wf = load_workflow_config(legacy_train_dag_path)
        train_model = train_wf.models[0]
        train_ctx = _build_context(train_wf, train_model, "local", "dev_test", str(artifacts_dir))
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        # Predict
        pred_wf = load_workflow_config(legacy_predict_dag_path)
        pred_model = pred_wf.models[0]
        pred_ctx = _build_context(pred_wf, pred_model, "local", "dev_test", str(artifacts_dir))

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
