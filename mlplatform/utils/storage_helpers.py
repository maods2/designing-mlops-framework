"""Helpers for persisting plots and HTML via the Storage interface."""

from __future__ import annotations

import io
from typing import Any, Union

from mlplatform.storage.base import Storage


def save_plot(fig: Any, path: str, storage: Storage) -> None:
    """Serialize figure to PNG and persist via storage."""
    buf = io.BytesIO()
    if hasattr(fig, "savefig"):
        fig.savefig(buf, format="png", bbox_inches="tight")
    elif hasattr(fig, "to_image"):
        buf.write(fig.to_image(format="png"))
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)!r}")
    buf.seek(0)
    storage.save(path, buf.getvalue())


def save_html(html: Union[str, bytes], path: str, storage: Storage) -> None:
    """Encode HTML and persist via storage."""
    data = html.encode("utf-8") if isinstance(html, str) else html
    storage.save(path, data)
