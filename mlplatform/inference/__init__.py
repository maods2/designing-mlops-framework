"""Inference strategies for prediction execution."""

from mlplatform.inference.base import InferenceStrategy
from mlplatform.inference.in_process import InProcessInference

__all__ = ["InferenceStrategy", "InProcessInference"]
