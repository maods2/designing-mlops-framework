"""Local filesystem storage backend."""

import pickle
from pathlib import Path

from mlops_framework.backends.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Filesystem-based storage using pickle for serialization."""
    
    def __init__(self, base_path: str = "./artifacts"):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, obj: object) -> None:
        """Save object as {name}.pkl in base_path."""
        path = self.base_path / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    
    def load(self, name: str) -> object:
        """Load object from {name}.pkl."""
        path = self.base_path / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
