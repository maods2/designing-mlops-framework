"""ExecutionContext - unified context passed to steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlplatform.config.schema import RunConfig
from mlplatform.etb.base import ExperimentTracker
from mlplatform.storage.base import Storage

if TYPE_CHECKING:
    from mlplatform.runners.base import Runner


@dataclass
class ExecutionContext:
    """Context passed to each step containing all execution primitives and config."""

    storage: Storage
    etb: ExperimentTracker
    runner: Runner
    run_config: RunConfig
    feature: str
    model_name: str
    version: str
    step_name: str
    custom: dict[str, Any]
