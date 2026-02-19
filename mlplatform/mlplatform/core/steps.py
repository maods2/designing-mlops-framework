"""Step abstractions - TrainStep, InferenceStep, PreprocessStep."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mlplatform.core.context import ExecutionContext


class Step(ABC):
    """Abstract base for pipeline steps."""

    @abstractmethod
    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        """Execute the step."""
        ...


class TrainStep(Step):
    """Base class for training steps. Uses storage, etb, and custom config."""

    def _artifact_path(self, name: str) -> str:
        """Resolve artifact path within feature/model/version hierarchy."""
        ctx = getattr(self, "_context", None)
        if ctx is None:
            raise RuntimeError("Context not set. Use within run() only.")
        return f"{ctx.feature}/{ctx.model_name}/{ctx.version}/{name}"

    def save_artifact(self, name: str, obj: Any) -> None:
        path = self._artifact_path(name)
        self._context.storage.save(path, obj)

    def load_artifact(self, name: str) -> Any:
        path = self._artifact_path(name)
        return self._context.storage.load(path)

    def log_params(self, params: dict[str, Any]) -> None:
        self._context.etb.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._context.etb.log_metrics(metrics)

    def log_artifact(self, path: str, artifact: Any) -> None:
        self._context.etb.log_artifact(path, artifact)

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclass must implement run()")


class InferenceStep(Step):
    """Base class for inference steps. Uses storage and custom config."""

    def _artifact_path(self, name: str) -> str:
        ctx = getattr(self, "_context", None)
        if ctx is None:
            raise RuntimeError("Context not set. Use within run() only.")
        return f"{ctx.feature}/{ctx.model_name}/{ctx.version}/{name}"

    def load_artifact(self, name: str) -> Any:
        path = self._artifact_path(name)
        return self._context.storage.load(path)

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclass must implement run()")


class PreprocessStep(Step):
    """Base class for preprocessing steps."""

    def _artifact_path(self, name: str) -> str:
        ctx = getattr(self, "_context", None)
        if ctx is None:
            raise RuntimeError("Context not set. Use within run() only.")
        return f"{ctx.feature}/{ctx.model_name}/{ctx.version}/{name}"

    def save_artifact(self, name: str, obj: Any) -> None:
        path = self._artifact_path(name)
        self._context.storage.save(path, obj)

    def load_artifact(self, name: str) -> Any:
        path = self._artifact_path(name)
        return self._context.storage.load(path)

    def run(self, context: ExecutionContext, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclass must implement run()")
