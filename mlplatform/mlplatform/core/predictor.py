"""BasePredictor - serving abstraction for inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    """Base class for prediction. Same core used across batch-local, online-REST, batch-Spark.

    Implementations must define:
    - load_model(): load model artifacts (called before predict)
    - predict(data): run prediction on a chunk of data

    Lifecycle hooks
    ---------------
    Override :meth:`setup` and/or :meth:`teardown` for custom initialization
    or cleanup.  The framework calls these automatically when orchestrating
    via ``run_workflow`` or invocation strategies.
    """

    def setup(self) -> None:
        """Called before :meth:`load_model`. Override for custom initialization.

        ``self.context`` is already set when this method is invoked.
        """

    @abstractmethod
    def load_model(self) -> Any:
        """Load model from storage. The predictor receives an ExecutionContext
        (set as self.context) providing access to storage and experiment tracking."""
        ...

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Run prediction on a chunk of data. Returns predictions."""
        ...

    def teardown(self) -> None:
        """Called after prediction is complete (even on error). Override for cleanup."""
