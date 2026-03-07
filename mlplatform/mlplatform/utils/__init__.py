"""Utility helpers for the mlplatform framework.

This subpackage provides reusable helpers that can be used across models and
pipelines regardless of which storage or tracking backend is in use.

**Install**::

    pip install mlplatform[utils]   # includes optional matplotlib dependency

The serialisation helpers (:func:`sanitize`, :func:`to_serializable`) have no
extra dependencies beyond the mlplatform base install.  The storage helpers
(:func:`save_plot`, :func:`save_html`) require any
:class:`~mlplatform.storage.base.Storage` implementation and, for plots,
``matplotlib`` or ``plotly``.

Quick-start example::

    from mlplatform.storage import LocalFileSystem
    from mlplatform.utils import sanitize, to_serializable, save_plot, save_html

    storage = LocalFileSystem("./artifacts")

    # Persist a matplotlib figure
    save_plot(fig, "reports/roc.png", storage)

    # Persist an HTML report
    save_html("<h1>Results</h1>", "reports/summary.html", storage)

    # Coerce metrics dict to JSON-safe types before logging
    clean = sanitize({"loss": float("nan"), "acc": 0.95})

    # Convert a dataclass or Pydantic model to a plain dict
    plain = to_serializable(my_metrics_object)
"""

from mlplatform.utils.logging import get_logger
from mlplatform.utils.serialization import sanitize, to_serializable
from mlplatform.utils.storage_helpers import save_html, save_plot

__all__ = [
    "get_logger",
    "sanitize",
    "to_serializable",
    "save_plot",
    "save_html",
]
