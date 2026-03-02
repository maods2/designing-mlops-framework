"""Unit tests for mlplatform.core.predictor (BasePredictor interface)."""

from __future__ import annotations

import pytest

from mlplatform.core.predictor import BasePredictor


class TestBasePredictorInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BasePredictor()  # type: ignore[abstract]

    def test_concrete_subclass_with_predict(self):
        class GoodPredictor(BasePredictor):
            def load_model(self):
                return None

            def predict(self, data):
                return data

        p = GoodPredictor()
        assert p.predict([1, 2, 3]) == [1, 2, 3]

    def test_missing_predict_raises(self):
        class BadPredictor(BasePredictor):
            def load_model(self):
                return None
            # predict is not implemented

        with pytest.raises(TypeError):
            BadPredictor()  # type: ignore[abstract]

    def test_missing_load_model_raises(self):
        class BadPredictor(BasePredictor):
            def predict(self, data):
                return data
            # load_model is not implemented

        with pytest.raises(TypeError):
            BadPredictor()  # type: ignore[abstract]

    def test_predict_method_name(self):
        """Ensure the method is named 'predict' (not 'predict_chunk')."""
        assert hasattr(BasePredictor, "predict")
        assert not hasattr(BasePredictor, "predict_chunk")
