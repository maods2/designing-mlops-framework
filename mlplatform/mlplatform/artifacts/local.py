"""Local JSON-based artifact store implementation."""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any

from mlplatform.artifacts.base import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    """JSON-file-backed artifact store for local development."""

    def __init__(self, base_path: str = "./artifacts") -> None:
        self.base_path = Path(base_path)
        self._registry_path = self.base_path / "model_registry.json"
        self._registry: dict[str, list[dict[str, Any]]] = self._load_registry()

    def _load_registry(self) -> dict[str, list[dict[str, Any]]]:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register_model(self, model_name: str, metadata: dict[str, Any]) -> None:
        if model_name not in self._registry:
            self._registry[model_name] = []
        self._registry[model_name].append(metadata)
        self._save_registry()

    def resolve_model(self, model_name: str, version: str) -> dict[str, Any]:
        entries = self._registry.get(model_name, [])
        if not entries:
            raise ValueError(f"No registered models for '{model_name}'")
        if version == "latest":
            return entries[-1]
        for entry in entries:
            if entry.get("version") == version:
                return entry
        raise ValueError(f"Version '{version}' not found for model '{model_name}'")
