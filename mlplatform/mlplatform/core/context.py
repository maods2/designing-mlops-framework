"""ExecutionContext - unified context passed to steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlplatform.artifacts.base import ArtifactStore
    from mlplatform.core.predictor import BasePredictor
    from mlplatform.core.trainer import BaseTrainer
    from mlplatform.etb.base import ExperimentTracker
    from mlplatform.invocation.base import InvocationStrategy
    from mlplatform.storage.base import Storage


@dataclass
class ExecutionContext:
    """Context injected after resolution. Does NOT expose runners.

    Contains infrastructure abstractions and ML core objects needed by steps.
    - experiment_tracker is injected only for training workloads
    - invocation_strategy is injected only for inference workloads
    - trainer / predictor are set based on workload type
    """

    storage: Storage
    artifact_store: ArtifactStore
    experiment_tracker: Optional[ExperimentTracker]
    invocation_strategy: Optional[InvocationStrategy]
    runtime_config: dict[str, Any] = field(default_factory=dict)
    environment_metadata: dict[str, Any] = field(default_factory=dict)
    trainer: Optional[BaseTrainer] = None
    predictor: Optional[BasePredictor] = None
