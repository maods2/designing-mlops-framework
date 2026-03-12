"""No-op storage when backend is not activated."""

from __future__ import annotations

import logging
from typing import Any

from mlplatform.storage.base import Storage

_logger = logging.getLogger(__name__)


class NoneStorage(Storage):
    """No-op storage when backend is not explicitly set to 'local' or 'gcs'.

    Logs a warning on save/load. Use backend='local' or backend='gcs' to enable
    actual persistence.
    """

    def save(self, path: str, obj: Any) -> None:
        _logger.warning(
            "Storage backend not activated: save(%r) is a no-op. "
            "Set backend='local' or backend='gcs' to persist artifacts.",
            path,
        )

    def load(self, path: str) -> Any:
        _logger.warning(
            "Storage backend not activated: load(%r) will raise. "
            "Set backend='local' or backend='gcs' to persist artifacts.",
            path,
        )
        raise FileNotFoundError(
            f"Artifact {path!r} not found: storage backend not activated. "
            "Set backend='local' or backend='gcs' in Artifact() or create_artifacts()."
        )

    def exists(self, path: str) -> bool:
        return False

    def list_artifacts(self, prefix: str = "") -> list[str]:
        return []

    def delete(self, path: str) -> None:
        _logger.warning(
            "Storage backend not activated: delete(%r) is a no-op. "
            "Set backend='local' or backend='gcs' to persist artifacts.",
            path,
        )
