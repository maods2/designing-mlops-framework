"""Local filesystem storage implementation."""

import joblib
from pathlib import Path
from typing import Any

from mlplatform.storage.base import Storage


class LocalFileSystem(Storage):
    """Local filesystem storage for artifacts."""

    def __init__(self, base_path: str = "./artifacts") -> None:
        self.base_path = Path(base_path)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path."""
        return self.base_path / path

    def save(self, path: str, obj: Any) -> None:
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, full_path)

    def load(self, path: str) -> Any:
        full_path = self._resolve_path(path)
        return joblib.load(full_path)
