"""Execution context for step-based pipelines."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from mlops_framework.backends.storage.base import StorageBackend
from mlops_framework.backends.tracking.base import TrackingBackend

if TYPE_CHECKING:
    from mlops_framework.core.run_context import RunContext


class ExecutionContext:
    """
    Execution context injected into every step.
    
    Holds storage, tracking, logger, config, and optional run_context.
    Everything depends on interfaces — not implementations.
    Enables local/cloud interchangeability.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        tracker: TrackingBackend,
        logger: Optional[Callable[[str], None]] = None,
        config: Optional[Dict[str, Any]] = None,
        run_context: Optional["RunContext"] = None,
    ):
        """
        Initialize the execution context.
        
        Args:
            storage: Storage backend for artifacts (save/load)
            tracker: Tracking backend for metrics and params
            logger: Logging callable (e.g. logging.info or print)
            config: Configuration dictionary for the run
            run_context: Optional runtime context (run_id, model_name, etc.)
        """
        self.storage = storage
        self.tracker = tracker
        self.logger = logger or print
        self.config = config or {}
        self.run_context = run_context
