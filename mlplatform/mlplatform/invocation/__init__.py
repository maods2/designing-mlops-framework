"""Invocation strategies for prediction execution."""

from mlplatform.invocation.base import InvocationStrategy
from mlplatform.invocation.in_process import InProcessInvocation

__all__ = [
    "InvocationStrategy",
    "InProcessInvocation",
]
