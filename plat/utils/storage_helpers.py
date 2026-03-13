"""Helpers for persisting plots and HTML reports via the Storage interface.

These helpers wrap serialized bytes with :meth:`Storage.save` so that the
artifacts are stored using the same backend and path conventions as every
other mlplatform artifact.  Retrieve them later with :meth:`Storage.load`
to get the raw ``bytes`` back.

Install ``mlplatform[utils]`` to get the optional ``matplotlib`` dependency
needed for :func:`save_plot`.
"""

from __future__ import annotations

import io
from typing import Any, Union

from mlplatform.storage.base import Storage


def save_plot(fig: Any, path: str, storage: Storage) -> None:
    """Serialize *fig* to PNG bytes and persist via *storage*.

    Supports:

    - **matplotlib** ``Figure`` objects (``fig.savefig``)
    - **plotly** ``Figure`` objects (``fig.to_image``)

    The artifact is stored as a joblib-wrapped ``bytes`` object; retrieve it
    with ``storage.load(path)`` to get back the raw PNG bytes.

    Args:
        fig: A matplotlib or plotly Figure instance.
        path: Destination path relative to the storage backend's base path.
        storage: Any :class:`~mlplatform.storage.base.Storage` implementation.

    Raises:
        TypeError: If *fig* is not a recognised figure type.

    Example::

        import matplotlib.pyplot as plt
        from mlplatform.storage import LocalFileSystem
        from mlplatform.utils import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_plot(fig, "reports/trend.png", LocalFileSystem("./artifacts"))
        plt.close(fig)
    """
    buf = io.BytesIO()

    if hasattr(fig, "savefig"):
        # matplotlib Figure
        fig.savefig(buf, format="png", bbox_inches="tight")
    elif hasattr(fig, "to_image"):
        # plotly Figure
        buf.write(fig.to_image(format="png"))
    else:
        raise TypeError(
            f"Unsupported figure type: {type(fig)!r}. "
            "Expected a matplotlib or plotly Figure."
        )

    storage.save(path, buf.getvalue())


def save_html(html: Union[str, bytes], path: str, storage: Storage) -> None:
    """Encode *html* and persist via *storage*.

    Accepts either a Unicode string or pre-encoded bytes.  The artifact is
    stored as a joblib-wrapped ``bytes`` object; retrieve it with
    ``storage.load(path)`` to get back the raw HTML bytes.

    Args:
        html: HTML content as a ``str`` or ``bytes``.
        path: Destination path relative to the storage backend's base path.
        storage: Any :class:`~mlplatform.storage.base.Storage` implementation.

    Example::

        from mlplatform.storage import LocalFileSystem
        from mlplatform.utils import save_html

        report = "<h1>Model Report</h1><p>Accuracy: 0.95</p>"
        save_html(report, "reports/summary.html", LocalFileSystem("./artifacts"))
    """
    data: bytes = html if isinstance(html, bytes) else html.encode("utf-8")
    storage.save(path, data)
