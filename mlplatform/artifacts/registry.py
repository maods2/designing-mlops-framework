"""ArtifactRegistry — save/load artifacts with path convention and format dispatch."""

from __future__ import annotations

import io
import json
from typing import Any

from mlplatform.storage.base import Storage

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_JSON_EXTS = (".json",)
_HTML_EXTS = (".html", ".htm")


def _ext(name: str) -> str:
    """Return lowercase extension including dot, or empty string."""
    if "." in name:
        return "." + name.rsplit(".", 1)[-1].lower()
    return ""


def _is_image_ext(name: str) -> bool:
    return _ext(name) in _IMAGE_EXTS


def _is_figure(obj: Any) -> bool:
    """True if obj is a matplotlib or plotly Figure."""
    return hasattr(obj, "savefig") or hasattr(obj, "to_image")


def _serialize_for_save(name: str, obj: Any, ext: str) -> tuple[Any, bool]:
    """Return (data, use_bytes). use_bytes=True means save as raw bytes."""
    if _is_figure(obj) and ext in _IMAGE_EXTS:
        buf = _figure_to_png(obj)
        return (buf.getvalue(), True)
    if isinstance(obj, dict) and ext in _JSON_EXTS:
        from mlplatform.utils.serialization import sanitize
        data = json.dumps(sanitize(obj), indent=2).encode("utf-8")
        return (data, True)
    if ext in _HTML_EXTS:
        data = obj.encode("utf-8") if isinstance(obj, str) else obj
        return (data, True)
    return (obj, False)


def _figure_to_png(obj: Any) -> io.BytesIO:
    buf = io.BytesIO()
    if hasattr(obj, "savefig"):
        obj.savefig(buf, format="png", bbox_inches="tight")
    elif hasattr(obj, "to_image"):
        buf.write(obj.to_image(format="png"))
    else:
        raise TypeError(f"Unsupported figure type: {type(obj)!r}")
    buf.seek(0)
    return buf


def _resolve_path(feature_name: str, model_name: str, version: str, name: str) -> str:
    """Build artifact path: {feature}/{model}/{version}/{name}."""
    return f"{feature_name}/{model_name}/{version}/{name}"


def _deserialize_for_load(name: str, raw: Any, ext: str) -> Any:
    """Deserialize raw storage output based on path extension."""
    if ext in _JSON_EXTS:
        data = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(data)
    return raw


class ArtifactRegistry:
    """Registry for saving and loading ML artifacts.

    Path convention: ``{feature_name}/{model_name}/{version}/{name}``

    Use only :meth:`save` and :meth:`load`. Format is inferred from path extension
    and object type.
    """

    def __init__(
        self,
        storage: Storage,
        feature_name: str,
        model_name: str,
        version: str,
    ) -> None:
        self._storage = storage
        self._feature_name = feature_name
        self._model_name = model_name
        self._version = version

    @property
    def storage(self) -> Storage:
        """Direct access to the underlying Storage backend."""
        return self._storage

    def save(self, name: str, obj: Any) -> None:
        """Save an object. Format inferred from path extension and object type.

        - ``.png``, ``.jpg``, ``.jpeg`` + Figure → PNG image
        - ``.json`` + dict → JSON (sanitized)
        - ``.html`` + str/bytes → raw HTML
        - ``.pkl``, ``.joblib`` or other → joblib (models, arbitrary objects)
        """
        path = _resolve_path(self._feature_name, self._model_name, self._version, name)
        ext = _ext(name)
        data, use_bytes = _serialize_for_save(name, obj, ext)
        if use_bytes:
            self._storage.save_bytes(path, data)
        else:
            self._storage.save(path, data)

    def load(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load an object. JSON files are parsed to dict; others returned as-is.

        Override *model_name*/*version* for cross-model loading (e.g., ensembles).
        """
        m = model_name or self._model_name
        v = version or self._version
        path = _resolve_path(self._feature_name, m, v, name)
        raw = self._storage.load(path)
        return _deserialize_for_load(name, raw, _ext(name))

    def resolve_path(self, name: str) -> str:
        """Return the full artifact path for *name*."""
        return _resolve_path(self._feature_name, self._model_name, self._version, name)

    def exists(self, name: str) -> bool:
        """Return True if artifact exists."""
        return self._storage.exists(self.resolve_path(name))
