"""Named profile factories for standard environments."""

from __future__ import annotations

from typing import Any

from mlplatform.artifacts.local import LocalArtifactStore
from mlplatform.core.enums import ExecutionTarget
from mlplatform.etb.local_json import LocalJsonTracker
from mlplatform.invocation.distributed import DistributedInvocation
from mlplatform.invocation.in_process import InProcessInvocation
from mlplatform.invocation.rest import RESTInvocation
from mlplatform.profiles.profile import Profile
from mlplatform.runners.local import LocalJobRunner, LocalServiceRunner
from mlplatform.runners.local_spark import LocalSparkJobRunner
from mlplatform.storage.local import LocalFileSystem


def _make_local(base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    return Profile(
        name="local",
        execution_target=ExecutionTarget.LOCAL,
        job_runner=LocalJobRunner(),
        service_runner=LocalServiceRunner(),
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        default_invocation_strategy=InProcessInvocation(),
    )


def _make_local_spark(base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    return Profile(
        name="local-spark",
        execution_target=ExecutionTarget.LOCAL,
        job_runner=LocalSparkJobRunner(direct=True, **kwargs),
        service_runner=LocalServiceRunner(),
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        default_invocation_strategy=DistributedInvocation(),
    )


def _make_cloud_batch(base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    from mlplatform.runners.dataproc_spark import DataprocJobRunner
    return Profile(
        name="cloud-batch",
        execution_target=ExecutionTarget.CLOUD,
        job_runner=DataprocJobRunner(**kwargs),
        service_runner=None,
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        default_invocation_strategy=DistributedInvocation(),
    )


def _make_cloud_online(base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    return Profile(
        name="cloud-online",
        execution_target=ExecutionTarget.CLOUD,
        job_runner=LocalJobRunner(),
        service_runner=LocalServiceRunner(),
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        default_invocation_strategy=RESTInvocation(),
    )


def _make_cloud_batch_emulated(base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    return Profile(
        name="cloud-batch-emulated",
        execution_target=ExecutionTarget.EMULATED_CLOUD,
        job_runner=LocalSparkJobRunner(direct=True, **kwargs),
        service_runner=None,
        storage=LocalFileSystem(base_path=base_path),
        artifact_store=LocalArtifactStore(base_path=base_path),
        experiment_tracker=LocalJsonTracker(base_path=base_path),
        default_invocation_strategy=DistributedInvocation(),
    )


_PROFILE_FACTORIES: dict[str, Any] = {
    "local": _make_local,
    "local-spark": _make_local_spark,
    "cloud-batch": _make_cloud_batch,
    "cloud-online": _make_cloud_online,
    "cloud-batch-emulated": _make_cloud_batch_emulated,
}


def get_profile(name: str, base_path: str = "./artifacts", **kwargs: Any) -> Profile:
    """Instantiate a named profile."""
    factory = _PROFILE_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown profile: {name}. Available: {list(_PROFILE_FACTORIES)}")
    return factory(base_path=base_path, **kwargs)


def list_profiles() -> list[str]:
    """Return available profile names."""
    return list(_PROFILE_FACTORIES.keys())
