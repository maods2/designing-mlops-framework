"""End-to-end tests: train example_model, then predict, verify artifacts and output."""

from __future__ import annotations

import pytest
import pandas as pd

from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.runner import _build_context


class TestTrainAndPredict:
    def test_full_train_then_predict_cycle(
        self,
        train_dag_path,
        predict_dag_path,
        sample_train_data,
        sample_inference_df,
        artifacts_dir,
    ):
        """Train then predict end-to-end with example_model (new pipeline schema)."""
        # --- Training ---
        train_pipeline = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, task_id="train_model", config_names=["global", "train-local"]
        )
        train_task = train_pipeline.tasks[0]
        train_ctx = _build_context(
            train_pipeline, train_task, "local", "e2e_v1", str(artifacts_dir)
        )
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        # Verify artifacts
        model_path = (
            artifacts_dir
            / train_pipeline.feature_name
            / train_task.model_name
            / "e2e_v1"
            / "model.pkl"
        )
        scaler_path = model_path.parent / "scaler.pkl"
        assert model_path.exists(), f"model.pkl not found: {model_path}"
        assert scaler_path.exists(), f"scaler.pkl not found: {scaler_path}"

        # --- Prediction ---
        pred_pipeline = ConfigLoaderFactory.load_pipeline_config(
            predict_dag_path, task_id="predict", config_names=["global", "predict-local"]
        )
        pred_task = pred_pipeline.tasks[0]
        pred_ctx = _build_context(
            pred_pipeline, pred_task, "local", "e2e_v1", str(artifacts_dir)
        )

        from example_model.predict import MyPredictor
        predictor = MyPredictor()
        predictor.context = pred_ctx
        predictor.load_model()
        result = predictor.predict(sample_inference_df)

        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert len(result) == len(sample_inference_df)
        assert result["prediction"].notna().all()

    def test_new_pipeline_format_end_to_end(
        self,
        train_dag_path,
        predict_dag_path,
        sample_train_data,
        sample_inference_df,
        artifacts_dir,
    ):
        """New flat pipeline schema works end-to-end."""
        # Train
        train_pipeline = ConfigLoaderFactory.load_pipeline_config(
            train_dag_path, config_names=["global", "train-local"]
        )
        train_task = next(t for t in train_pipeline.tasks if t.module)
        train_ctx = _build_context(
            train_pipeline, train_task, "local", "e2e_new", str(artifacts_dir)
        )
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        # Verify model artifact
        model_path = (
            artifacts_dir
            / train_pipeline.feature_name
            / train_task.model_name
            / "e2e_new"
            / "model.pkl"
        )
        assert model_path.exists()

        # Predict
        pred_pipeline = ConfigLoaderFactory.load_pipeline_config(
            predict_dag_path, config_names=["global", "predict-local"]
        )
        pred_task = pred_pipeline.tasks[0]
        pred_ctx = _build_context(
            pred_pipeline, pred_task, "local", "e2e_new", str(artifacts_dir)
        )

        from example_model.predict import MyPredictor
        predictor = MyPredictor()
        predictor.context = pred_ctx
        predictor.load_model()
        result = predictor.predict(sample_inference_df)

        assert "prediction" in result.columns
        assert len(result) == len(sample_inference_df)
