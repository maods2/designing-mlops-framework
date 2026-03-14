"""Runner — execute pipelines from PipelineConfig.

Re-exports the main entry points so ``from mlplatform.runner import …`` works.
"""

from mlplatform.runner.dev import dev_context, dev_predict, dev_train
from mlplatform.runner.execute import execute
from mlplatform.runner.resolve import resolve_class

__all__ = [
    "execute",
    "dev_train",
    "dev_predict",
    "dev_context",
    "resolve_class",
]
