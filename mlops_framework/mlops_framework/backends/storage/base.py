"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """
    Abstract base class for artifact storage.
    
    Implementations: LocalStorage (filesystem), GCSStorage (Google Cloud).
    Run-scoping is handled by the runner via base_path or blob prefix.
    """
    
    @abstractmethod
    def save(self, name: str, obj: object) -> None:
        """
        Save an artifact.
        
        Args:
            name: Artifact name (used as filename/key)
            obj: Object to save (must be serializable)
        """
        pass
    
    @abstractmethod
    def load(self, name: str) -> object:
        """
        Load an artifact.
        
        Args:
            name: Artifact name
            
        Returns:
            Loaded object
        """
        pass
