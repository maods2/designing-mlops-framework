"""Artifact manager for versioned storage and retrieval."""

import os
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Dict
from mlops_framework.artifacts.types import ArtifactType


class ArtifactManager:
    """
    Manages artifact versioning and storage.
    
    Artifacts are stored per run with semantic types for easy retrieval.
    """
    
    def __init__(self, base_path: str, run_id: str, tracking_backend=None):
        """
        Initialize the artifact manager.
        
        Args:
            base_path: Base directory for artifact storage
            run_id: Current run ID
            tracking_backend: Optional tracking backend for logging artifacts
        """
        self.base_path = Path(base_path)
        self.run_id = run_id
        self.tracking_backend = tracking_backend
        self.run_artifact_path = self.base_path / run_id
        self.run_artifact_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, artifact: Any, name: str, artifact_type: ArtifactType = ArtifactType.OTHER) -> str:
        """
        Save an artifact to disk.
        
        Args:
            artifact: The artifact to save (model, data, etc.)
            name: Name for the artifact
            artifact_type: Semantic type of the artifact
            
        Returns:
            Path where the artifact was saved
        """
        # Create type-specific directory
        type_dir = self.run_artifact_path / artifact_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension based on artifact type
        if artifact_type == ArtifactType.METADATA:
            extension = '.json'
        else:
            extension = '.pkl'
        
        artifact_path = type_dir / f"{name}{extension}"
        
        # Save based on artifact type
        if artifact_type == ArtifactType.METADATA:
            # Use JSON for metadata
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2)
        else:
            # Use pickle for models and other artifacts
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        
        # Log to tracking backend if available
        if self.tracking_backend:
            relative_path = f"{artifact_type.value}/{artifact_path.name}"
            self.tracking_backend.log_artifact(str(artifact_path), relative_path)
        
        return str(artifact_path)
    
    def load(self, name: str, artifact_type: ArtifactType = ArtifactType.OTHER) -> Any:
        """
        Load an artifact from disk.
        
        Args:
            name: Name of the artifact
            artifact_type: Semantic type of the artifact
            
        Returns:
            Loaded artifact
        """
        type_dir = self.run_artifact_path / artifact_type.value
        
        # Try different extensions
        for ext in ['.pkl', '.json', '']:
            artifact_path = type_dir / f"{name}{ext}"
            if artifact_path.exists():
                if artifact_path.suffix == '.json':
                    with open(artifact_path, 'r') as f:
                        return json.load(f)
                else:
                    with open(artifact_path, 'rb') as f:
                        return pickle.load(f)
        
        raise FileNotFoundError(
            f"Artifact '{name}' of type '{artifact_type.value}' not found in run '{self.run_id}'"
        )
    
    def save_file(self, local_path: str, name: str, artifact_type: ArtifactType = ArtifactType.OTHER) -> str:
        """
        Save a file as an artifact (copy to artifact storage).
        
        Args:
            local_path: Path to the local file
            name: Name for the artifact
            artifact_type: Semantic type of the artifact
            
        Returns:
            Path where the artifact was saved
        """
        import shutil
        
        type_dir = self.run_artifact_path / artifact_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = type_dir / name
        shutil.copy2(local_path, artifact_path)
        
        # Log to tracking backend if available
        if self.tracking_backend:
            relative_path = f"{artifact_type.value}/{artifact_path.name}"
            self.tracking_backend.log_artifact(str(artifact_path), relative_path)
        
        return str(artifact_path)
    
    def get_artifact_path(self, name: str, artifact_type: ArtifactType = ArtifactType.OTHER) -> str:
        """
        Get the path to an artifact without loading it.
        
        Args:
            name: Name of the artifact
            artifact_type: Semantic type of the artifact
            
        Returns:
            Path to the artifact
        """
        type_dir = self.run_artifact_path / artifact_type.value
        
        # Try different extensions
        for ext in ['.pkl', '.json', '']:
            artifact_path = type_dir / f"{name}{ext}"
            if artifact_path.exists():
                return str(artifact_path)
        
        raise FileNotFoundError(
            f"Artifact '{name}' of type '{artifact_type.value}' not found in run '{self.run_id}'"
        )
    
    def list_artifacts(self, artifact_type: Optional[ArtifactType] = None) -> Dict[str, list]:
        """
        List all artifacts in the current run.
        
        Args:
            artifact_type: Optional filter by artifact type
            
        Returns:
            Dictionary mapping artifact types to lists of artifact names
        """
        artifacts = {}
        
        if artifact_type:
            types_to_check = [artifact_type]
        else:
            types_to_check = list(ArtifactType)
        
        for atype in types_to_check:
            type_dir = self.run_artifact_path / atype.value
            if type_dir.exists():
                artifacts[atype.value] = [
                    f.stem for f in type_dir.iterdir() if f.is_file()
                ]
        
        return artifacts
