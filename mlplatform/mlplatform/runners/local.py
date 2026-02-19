"""Local in-process runner implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.runners.base import Runner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class LocalRunner(Runner):
    """Execute steps in-process on the local machine."""

    def run(self, step: "Step", context: "ExecutionContext", **kwargs: Any) -> Any:
        return step.run(context, **kwargs)
