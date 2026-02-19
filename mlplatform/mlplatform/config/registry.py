"""Registry for pluggable Runner, Storage, ETB, and ArtifactStore backends."""

from __future__ import annotations

from typing import Any, Type

from mlplatform.artifacts.base import ArtifactStore
from mlplatform.artifacts.local import LocalArtifactStore
from mlplatform.etb.base import ExperimentTracker
from mlplatform.etb.local_json import LocalJsonTracker
from mlplatform.etb.none import NoneTracker
from mlplatform.runners.base import JobRunner, ServiceRunner
from mlplatform.runners.dataproc_spark import DataprocJobRunner
from mlplatform.runners.local import LocalJobRunner, LocalServiceRunner
from mlplatform.runners.local_spark import LocalSparkJobRunner
from mlplatform.storage.base import Storage
from mlplatform.storage.local import LocalFileSystem

JOB_RUNNER_REGISTRY: dict[str, Type[JobRunner]] = {
    "LocalJobRunner": LocalJobRunner,
    "LocalSparkJobRunner": LocalSparkJobRunner,
    "DataprocJobRunner": DataprocJobRunner,
}

SERVICE_RUNNER_REGISTRY: dict[str, Type[ServiceRunner]] = {
    "LocalServiceRunner": LocalServiceRunner,
}

STORAGE_REGISTRY: dict[str, Type[Storage]] = {
    "LocalFileSystem": LocalFileSystem,
}

ETB_REGISTRY: dict[str, Type[ExperimentTracker]] = {
    "NoneTracker": NoneTracker,
    "LocalJsonTracker": LocalJsonTracker,
}

ARTIFACT_STORE_REGISTRY: dict[str, Type[ArtifactStore]] = {
    "LocalArtifactStore": LocalArtifactStore,
}


def get_job_runner(name: str, **kwargs: Any) -> JobRunner:
    cls = JOB_RUNNER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown job runner: {name}. Available: {list(JOB_RUNNER_REGISTRY)}")
    return cls(**kwargs)


def get_service_runner(name: str, **kwargs: Any) -> ServiceRunner:
    cls = SERVICE_RUNNER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown service runner: {name}. Available: {list(SERVICE_RUNNER_REGISTRY)}")
    return cls(**kwargs)


def get_storage(name: str, **kwargs: Any) -> Storage:
    cls = STORAGE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown storage: {name}. Available: {list(STORAGE_REGISTRY)}")
    return cls(**kwargs)


def get_etb(name: str, **kwargs: Any) -> ExperimentTracker:
    cls = ETB_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown ETB: {name}. Available: {list(ETB_REGISTRY)}")
    return cls(**kwargs)


def get_artifact_store(name: str, **kwargs: Any) -> ArtifactStore:
    cls = ARTIFACT_STORE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown artifact store: {name}. Available: {list(ARTIFACT_STORE_REGISTRY)}")
    return cls(**kwargs)
