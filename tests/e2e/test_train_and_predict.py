"""End-to-end tests: train example_model, then predict, verify artifacts."""

from __future__ import annotations

import pytest
import pandas as pd

from mlplatform.config.loader import load_workflow_config
from mlplatform.runner import _build_context, run_workflow


class TestTrainAndPredict:
    def test_full_train_then_predict_cycle(
        self,
        legacy_train_dag_path,
        legacy_predict_dag_path,
        sample_train_data,
        sample_inference_df,
        artifacts_dir,
    ):
        """Train then predict end-to-end with example_model."""
        # --- Training ---
        train_wf = load_workflow_config(legacy_train_dag_path)
        train_model = train_wf.models[0]
        train_ctx = _build_context(
            train_wf, train_model, "local", "e2e_v1", str(artifacts_dir)
        )
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        # Verify model artifact exists
        model_path = (
            artifacts_dir
            / train_wf.feature_name
            / train_model.model_name
            / "e2e_v1"
            / "model.pkl"
        )
        assert model_path.exists(), f"model.pkl not found at {model_path}"
        scaler_path = (
            artifacts_dir
            / train_wf.feature_name
            / train_model.model_name
            / "e2e_v1"
            / "scaler.pkl"
        )
        assert scaler_path.exists(), f"scaler.pkl not found at {scaler_path}"

        # --- Prediction ---
        pred_wf = load_workflow_config(legacy_predict_dag_path)
        pred_model = pred_wf.models[0]
        pred_ctx = _build_context(
            pred_wf, pred_model, "local", "e2e_v1", str(artifacts_dir)
        )

        from example_model.predict import MyPredictor
        predictor = MyPredictor()
        predictor.context = pred_ctx
        predictor.load_model()
        result = predictor.predict(sample_inference_df)

        # Verify output
        assert isinstance(result, pd.DataFrame), "predict() must return a DataFrame"
        assert "prediction" in result.columns, "Output must contain 'prediction' column"
        assert len(result) == len(sample_inference_df), "Row count must be preserved"
        assert result["prediction"].notna().all(), "No NaN predictions"

    def test_new_dag_format_train(
        self,
        train_dag_path,
        sample_train_data,
        artifacts_dir,
    ):
        """New DAG format (with resources block) produces a working training run."""
        train_wf = load_workflow_config(train_dag_path)
        train_model = train_wf.models[0]
        train_ctx = _build_context(
            train_wf, train_model, "local", "e2e_newdag", str(artifacts_dir)
        )
        train_ctx.optional_configs["train_data"] = sample_train_data

        from example_model.train import MyTrainer
        trainer = MyTrainer()
        trainer.context = train_ctx
        trainer.train()

        model_path = (
            artifacts_dir
            / train_wf.feature_name
            / train_model.model_name
            / "e2e_newdag"
            / "model.pkl"
        )
        assert model_path.exists()
