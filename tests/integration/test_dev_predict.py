"""Integration tests for dev_predict."""

from __future__ import annotations

import pytest
import pandas as pd

from mlplatform.runner import _build_context
from mlplatform.config.loader import load_workflow_config


class TestDevPredict:
    def test_predict_with_data(self, legacy_train_dag_path, legacy_predict_dag_path,
                               sample_train_data, sample_inference_df, artifacts_dir):
        """Train a model, then run predict with explicit data."""
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

    def test_predict_method_name(self, legacy_predict_dag_path, legacy_train_dag_path,
                                  sample_train_data, artifacts_dir):
        """Predictor must expose predict() not predict_chunk()."""
        from example_model.predict import MyPredictor
        assert hasattr(MyPredictor, "predict")
        assert not hasattr(MyPredictor, "predict_chunk")
