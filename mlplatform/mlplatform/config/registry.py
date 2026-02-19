"""Registry for pluggable Runner, Storage, and ETB backends."""

from __future__ import annotations

from typing import Any, Type

from mlplatform.etb.base import ExperimentTracker
from mlplatform.etb.local_json import LocalJsonTracker
from mlplatform.etb.none import NoneTracker
from mlplatform.runners.base import Runner
from mlplatform.runners.dataproc_spark import DataprocSparkRunner
from mlplatform.runners.local import LocalRunner
from mlplatform.runners.local_spark import LocalSparkRunner
from mlplatform.storage.base import Storage
from mlplatform.storage.local import LocalFileSystem

RUNNER_REGISTRY: dict[str, Type[Runner]] = {
    "LocalRunner": LocalRunner,
    "LocalSparkRunner": LocalSparkRunner,
    "DataprocSparkRunner": DataprocSparkRunner,
}

STORAGE_REGISTRY: dict[str, Type[Storage]] = {
    "LocalFileSystem": LocalFileSystem,
}

ETB_REGISTRY: dict[str, Type[ExperimentTracker]] = {
    "NoneTracker": NoneTracker,
    "LocalJsonTracker": LocalJsonTracker,
}


def get_runner(name: str, **kwargs: Any) -> Runner:
    """Instantiate a runner by name."""
    cls = RUNNER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNER_REGISTRY)}")
    return cls(**kwargs)


def get_storage(name: str, **kwargs: Any) -> Storage:
    """Instantiate storage by name."""
    cls = STORAGE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown storage: {name}. Available: {list(STORAGE_REGISTRY)}")
    return cls(**kwargs)


def get_etb(name: str, **kwargs: Any) -> ExperimentTracker:
    """Instantiate experiment tracker by name."""
    cls = ETB_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown ETB: {name}. Available: {list(ETB_REGISTRY)}")
    return cls(**kwargs)
