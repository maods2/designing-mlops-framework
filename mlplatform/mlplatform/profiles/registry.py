"""Profile definitions and registry.

A Profile is a declarative infrastructure bundle that determines which Storage,
ExperimentTracker, and InvocationStrategy are used for a given environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mlplatform.invocation.base import InvocationStrategy
from mlplatform.invocation.fastapi_serving import FastAPIInvocation
from mlplatform.invocation.in_process import InProcessInvocation
from mlplatform.invocation.spark_batch import SparkBatchInvocation
from mlplatform.storage.base import Storage
from mlplatform.storage.gcs import GCSStorage
from mlplatform.storage.local import LocalFileSystem
from mlplatform.tracking.base import ExperimentTracker
from mlplatform.tracking.local import LocalJsonTracker
from mlplatform.tracking.none import NoneTracker
from mlplatform.tracking.vertexai import VertexAITracker


@dataclass
class Profile:
    """Declarative infrastructure bundle for a given execution environment."""

    name: str
    storage_factory: Callable[[str], Storage]
    tracker_factory: Callable[..., ExperimentTracker]  # (base_path, metrics_path=None)
    invocation_strategy_factory: Callable[[], InvocationStrategy]
    extra: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, Profile] = {}


def register_profile(profile: Profile) -> None:
    """Register a profile by name."""
    _REGISTRY[profile.name] = profile


def get_profile(name: str) -> Profile:
    """Look up a registered profile by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Local profiles (no cloud dependencies)
# ---------------------------------------------------------------------------

def _local_tracker_factory(base_path: str, metrics_path: str | None = None) -> ExperimentTracker:
    return LocalJsonTracker(base_path, metrics_path=metrics_path)


register_profile(Profile(
    name="local",
    storage_factory=lambda bp: LocalFileSystem(bp),
    tracker_factory=_local_tracker_factory,
    invocation_strategy_factory=lambda: InProcessInvocation(),
))

register_profile(Profile(
    name="local-spark",
    storage_factory=lambda bp: LocalFileSystem(bp),
    tracker_factory=_local_tracker_factory,
    invocation_strategy_factory=lambda: SparkBatchInvocation(),
))

register_profile(Profile(
    name="cloud-batch-emulated",
    storage_factory=lambda bp: LocalFileSystem(bp),
    tracker_factory=_local_tracker_factory,
    invocation_strategy_factory=lambda: SparkBatchInvocation(),
))

# ---------------------------------------------------------------------------
# Cloud profiles (require google-cloud-* SDKs at runtime)
# ---------------------------------------------------------------------------

def _vertex_tracker_factory(base_path: str, metrics_path: str | None = None) -> ExperimentTracker:
    return VertexAITracker(experiment_name=base_path)


register_profile(Profile(
    name="cloud-batch",
    storage_factory=lambda bp: GCSStorage(bp),
    tracker_factory=_vertex_tracker_factory,
    invocation_strategy_factory=lambda: SparkBatchInvocation(),
))

register_profile(Profile(
    name="cloud-online",
    storage_factory=lambda bp: GCSStorage(bp),
    tracker_factory=_vertex_tracker_factory,
    invocation_strategy_factory=lambda: FastAPIInvocation(),
))

register_profile(Profile(
    name="cloud-train",
    storage_factory=lambda bp: GCSStorage(bp),
    tracker_factory=_vertex_tracker_factory,
    invocation_strategy_factory=lambda: InProcessInvocation(),
))
