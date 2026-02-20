"""BaseTrainer - training abstraction contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base class for model training. Implement train() with your training logic.

    The trainer receives an ExecutionContext (set as self.context) before train()
    is called, providing access to storage, experiment_tracker, and optional_configs.
    """

    @abstractmethod
    def train(self) -> None:
        """Execute the training workflow."""
        ...
