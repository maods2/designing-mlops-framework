"""Profile definitions and registry.

A Profile is a declarative infrastructure bundle that determines which Storage,
ExperimentTracker, and InferenceStrategy are used for a given environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mlplatform.inference.base import InferenceStrategy
from mlplatform.storage.base import Storage
from mlplatform.tracking.base import ExperimentTracker


@dataclass
class Profile:
    """Declarative infrastructure bundle for a given execution environment."""

    name: str
    storage_factory: Callable[[str, dict[str, Any]], Storage]
    tracker_factory: Callable[[str, dict[str, Any]], ExperimentTracker]
    inference_strategy_factory: Callable[[], InferenceStrategy]
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
# Lazy factory helpers
# ---------------------------------------------------------------------------

def _local_storage(bp: str, extra: dict[str, Any]) -> Storage:
    from mlplatform.storage.local import LocalFileSystem
    return LocalFileSystem(bp)


def _gcs_storage(bp: str, extra: dict[str, Any]) -> Storage:
    from mlplatform.storage.gcs import GCSStorage
    return GCSStorage(bp, project=extra.get("gcp_project"))


def _local_json_tracker(bp: str, extra: dict[str, Any]) -> ExperimentTracker:
    from mlplatform.tracking.local import LocalJsonTracker
    return LocalJsonTracker(bp)


def _none_tracker(bp: str, extra: dict[str, Any]) -> ExperimentTracker:
    from mlplatform.tracking.none import NoneTracker
    return NoneTracker()


def _vertex_tracker(bp: str, extra: dict[str, Any]) -> ExperimentTracker:
    from mlplatform.tracking.vertexai import VertexAITracker
    return VertexAITracker(
        experiment_name=bp,
        project=extra.get("gcp_project"),
        location=extra.get("gcp_location", "us-central1"),
    )


def _in_process_inference() -> InferenceStrategy:
    from mlplatform.inference.in_process import InProcessInference
    return InProcessInference()


def _spark_batch_inference() -> InferenceStrategy:
    from mlplatform.inference.spark_batch import SparkBatchInference
    return SparkBatchInference()


def _fastapi_inference() -> InferenceStrategy:
    from mlplatform.inference.fastapi_serving import FastAPIInference
    return FastAPIInference()


# ---------------------------------------------------------------------------
# Local profiles
# ---------------------------------------------------------------------------

register_profile(Profile(
    name="local",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    inference_strategy_factory=_in_process_inference,
))

register_profile(Profile(
    name="local-spark",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    inference_strategy_factory=_spark_batch_inference,
))

register_profile(Profile(
    name="cloud-batch-emulated",
    storage_factory=_local_storage,
    tracker_factory=_local_json_tracker,
    inference_strategy_factory=_spark_batch_inference,
))

# ---------------------------------------------------------------------------
# Cloud profiles
# ---------------------------------------------------------------------------

import os as _os

_GCP_DEFAULTS: dict[str, Any] = {
    "gcp_project": _os.environ.get("GOOGLE_CLOUD_PROJECT"),
    "gcp_location": _os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
}

register_profile(Profile(
    name="cloud-batch",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    inference_strategy_factory=_spark_batch_inference,
    extra=_GCP_DEFAULTS,
))

register_profile(Profile(
    name="cloud-online",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    inference_strategy_factory=_fastapi_inference,
    extra=_GCP_DEFAULTS,
))

register_profile(Profile(
    name="cloud-train",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    inference_strategy_factory=_in_process_inference,
    extra=_GCP_DEFAULTS,
))
