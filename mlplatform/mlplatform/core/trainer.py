"""BaseTrainer - training abstraction contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base class for model training. Implement train() with your training logic.

    The trainer receives an ExecutionContext (set as self.context) before train()
    is called, providing access to storage, experiment_tracker, and optional_configs.

    Lifecycle hooks
    ---------------
    Override :meth:`setup` and/or :meth:`teardown` for custom initialization
    or cleanup that should run before/after :meth:`train`.  The framework
    calls these automatically when orchestrating via ``run_workflow``.
    """

    def setup(self) -> None:
        """Called before :meth:`train`. Override for custom initialization.

        ``self.context`` is already set when this method is invoked.
        """

    @abstractmethod
    def train(self) -> None:
        """Execute the training workflow."""
        ...

    def teardown(self) -> None:
        """Called after :meth:`train` (even if train raised). Override for cleanup."""
