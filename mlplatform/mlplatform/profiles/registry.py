"""Profile definitions and registry.

A Profile is a declarative infrastructure bundle that determines which Storage,
ExperimentTracker, and InvocationStrategy are used for a given environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mlplatform.invocation.base import InvocationStrategy
from mlplatform.storage.base import Storage
from mlplatform.tracking.base import ExperimentTracker


@dataclass
class Profile:
    """Declarative infrastructure bundle for a given execution environment."""

    name: str
    storage_factory: Callable[[str], Storage]
    tracker_factory: Callable[[str], ExperimentTracker]
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
# Lazy factory helpers — cloud dependencies are imported only when the
# factory is actually *called*, not when the module is loaded.
# ---------------------------------------------------------------------------

def _local_storage(bp: str) -> Storage:
    from mlplatform.storage.local import LocalFileSystem
    return LocalFileSystem(bp)


def _gcs_storage(bp: str) -> Storage:
    from mlplatform.storage.gcs import GCSStorage
    return GCSStorage(bp)


def _local_json_tracker(bp: str) -> ExperimentTracker:
    from mlplatform.tracking.local import LocalJsonTracker
    return LocalJsonTracker(bp)


def _none_tracker(bp: str) -> ExperimentTracker:
    from mlplatform.tracking.none import NoneTracker
    return NoneTracker()


def _vertex_tracker(bp: str) -> ExperimentTracker:
    from mlplatform.tracking.vertexai import VertexAITracker
    return VertexAITracker(experiment_name=bp)


def _in_process_invocation() -> InvocationStrategy:
    from mlplatform.invocation.in_process import InProcessInvocation
    return InProcessInvocation()


def _spark_batch_invocation() -> InvocationStrategy:
    from mlplatform.invocation.spark_batch import SparkBatchInvocation
    return SparkBatchInvocation()


def _fastapi_invocation() -> InvocationStrategy:
    from mlplatform.invocation.fastapi_serving import FastAPIInvocation
    return FastAPIInvocation()


# ---------------------------------------------------------------------------
# Local profiles (no cloud dependencies)
# ---------------------------------------------------------------------------

register_profile(Profile(
    name="local",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    invocation_strategy_factory=_in_process_invocation,
))

register_profile(Profile(
    name="local-spark",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    invocation_strategy_factory=_spark_batch_invocation,
))

register_profile(Profile(
    name="cloud-batch-emulated",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    invocation_strategy_factory=_spark_batch_invocation,
))

# ---------------------------------------------------------------------------
# Cloud profiles (require google-cloud-* SDKs at runtime)
# ---------------------------------------------------------------------------

register_profile(Profile(
    name="cloud-batch",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    invocation_strategy_factory=_spark_batch_invocation,
))

register_profile(Profile(
    name="cloud-online",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    invocation_strategy_factory=_fastapi_invocation,
))

register_profile(Profile(
    name="cloud-train",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    invocation_strategy_factory=_in_process_invocation,
))
