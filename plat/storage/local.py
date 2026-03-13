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

    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()

    def list_artifacts(self, prefix: str = "") -> list[str]:
        base = self._resolve_path(prefix) if prefix else self.base_path
        if not base.exists():
            return []
        return [
            str(p.relative_to(self.base_path))
            for p in base.rglob("*")
            if p.is_file()
        ]

    def delete(self, path: str) -> None:
        full_path = self._resolve_path(path)
        if full_path.exists():
            full_path.unlink()
