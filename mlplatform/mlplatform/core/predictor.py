"""BasePredictor - serving abstraction for inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mlplatform.storage.base import Storage


class BasePredictor(ABC):
    """Base class for prediction. Same core used across BatchLocal, OnlineREST, BatchSpark."""

    @abstractmethod
    def load_model(self, storage: Storage, path: str) -> Any:
        """Load model from storage."""
        ...

    @abstractmethod
    def predict_chunk(self, data: Any) -> Any:
        """Run prediction on a chunk of data. Returns predictions."""
        ...

    def run(
        self,
        data: Any,
        storage: Storage | None = None,
        model_path: str | None = None,
    ) -> Any:
        """BatchLocal entry point: load model and predict. Override load_model/predict_chunk."""
        if storage is None or model_path is None:
            raise ValueError("storage and model_path required for run()")
        model = self.load_model(storage, model_path)
        return self.predict_chunk(data)
