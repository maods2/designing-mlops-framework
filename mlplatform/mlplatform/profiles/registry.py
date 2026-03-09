"""Profile definitions and registry.

A Profile is a declarative infrastructure bundle that determines which Storage,
ExperimentTracker, and InvocationStrategy are used for a given environment.

The ``extra`` dict is forwarded to every factory so that environment-specific
settings (e.g. ``gcp_project``, ``gcp_location``) can be injected without
changing factory signatures:

.. code-block:: python

    Profile(
        name="gcp-local",
        storage_factory=_gcs_storage,
        tracker_factory=_vertex_tracker,
        invocation_strategy_factory=_in_process_invocation,
        extra={"gcp_project": "my-dev-project"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mlplatform.invocation.base import InvocationStrategy
from mlplatform.storage.base import Storage
from mlplatform.tracking.base import ExperimentTracker


@dataclass
class Profile:
    """Declarative infrastructure bundle for a given execution environment.

    ``storage_factory`` and ``tracker_factory`` both receive ``(base_path,
    extra)`` so they can consume profile-level settings such as ``gcp_project``
    or ``gcp_location``.
    """

    name: str
    storage_factory: Callable[[str, dict[str, Any]], Storage]
    tracker_factory: Callable[[str, dict[str, Any]], ExperimentTracker]
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
#
# ``gcp_project`` defaults to the GOOGLE_CLOUD_PROJECT env var so that local
# runs only need `export GOOGLE_CLOUD_PROJECT=my-project` (or a .env file).
# On GCP (Vertex AI, Cloud Run, GCE) the value is auto-detected via ADC and
# this env var is not required.
#
# To pin a project explicitly when registering your own profile:
#
#   from mlplatform.profiles import register_profile, Profile
#   register_profile(Profile(
#       name="gcp-local",
#       storage_factory=_gcs_storage,
#       tracker_factory=_vertex_tracker,
#       invocation_strategy_factory=_in_process_invocation,
#       extra={"gcp_project": "my-dev-project"},
#   ))
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
    invocation_strategy_factory=_spark_batch_invocation,
    extra=_GCP_DEFAULTS,
))

register_profile(Profile(
    name="cloud-online",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    invocation_strategy_factory=_fastapi_invocation,
    extra=_GCP_DEFAULTS,
))

register_profile(Profile(
    name="cloud-train",
    storage_factory=_gcs_storage,
    tracker_factory=_vertex_tracker,
    invocation_strategy_factory=_in_process_invocation,
    extra=_GCP_DEFAULTS,
))
