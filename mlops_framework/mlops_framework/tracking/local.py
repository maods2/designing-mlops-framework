"""Local filesystem-based tracking backend."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from mlops_framework.tracking.interface import TrackingBackend


class LocalTrackingBackend(TrackingBackend):
    """
    Filesystem-based tracking backend.
    
    Stores runs in a directory structure:
    runs/
      {run_id}/
        params.json
        metrics.json
        artifacts/
        models/
        tags.json
    """
    
    def __init__(self, base_path: str = "./runs", run_id: Optional[str] = None):
        """
        Initialize the local tracking backend.
        
        Args:
            base_path: Base directory for storing runs
            run_id: Optional run ID (auto-generated if None)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = self._generate_run_id()
        
        self.run_id = run_id
        self.run_path = self.base_path / run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata files
        self.params_file = self.run_path / "params.json"
        self.metrics_file = self.run_path / "metrics.json"
        self.tags_file = self.run_path / "tags.json"
        self.artifacts_dir = self.run_path / "artifacts"
        self.models_dir = self.run_path / "models"
        
        self.artifacts_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing data
        self.params = self._load_json(self.params_file, {})
        self.metrics = self._load_json(self.metrics_file, {})
        self.tags = self._load_json(self.tags_file, {})
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"run_{timestamp}"
    
    def _load_json(self, file_path: Path, default: Dict) -> Dict:
        """Load JSON file or return default if it doesn't exist."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    
    def _save_json(self, file_path: Path, data: Dict) -> None:
        """Save dictionary to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter."""
        # Convert value to JSON-serializable format
        if not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        
        self.params[name] = value
        self._save_json(self.params_file, self.params)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {"value": value}
        if step is not None:
            metric_entry["step"] = step
        metric_entry["timestamp"] = datetime.now().isoformat()
        
        self.metrics[name].append(metric_entry)
        self._save_json(self.metrics_file, self.metrics)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact by copying it to the run directory."""
        import shutil
        
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")
        
        if artifact_path is None:
            artifact_path = source.name
        
        dest = self.artifacts_dir / artifact_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        if source.is_file():
            shutil.copy2(source, dest)
        else:
            # For directories, copy recursively
            shutil.copytree(source, dest, dirs_exist_ok=True)
    
    def log_model(self, model: Any, artifact_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a trained model."""
        import pickle
        
        model_path = self.models_dir / artifact_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = model_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current run."""
        self.tags.update(tags)
        self._save_json(self.tags_file, self.tags)
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id
