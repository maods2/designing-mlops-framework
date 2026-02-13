"""Local filesystem tracking backend for step-based flow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mlops_framework.backends.tracking.base import TrackingBackend


class LocalTracker(TrackingBackend):
    """
    Local tracking backend for metrics and params.
    
    Persists to runs/{run_id}/ or keeps in-memory if no base_path.
    Compatible with step-based ExecutionContext.
    """
    
    def __init__(self, base_path: str = "./runs", run_id: Optional[str] = None):
        """
        Initialize local tracker.
        
        Args:
            base_path: Base directory for runs
            run_id: Optional run ID (auto-generated if None)
        """
        self.base_path = Path(base_path)
        self._run_id = run_id or self._generate_run_id()
        self._params: dict = {}
        self._metrics: dict = {}
        
        if base_path:
            self.run_path = self.base_path / self._run_id
            self.run_path.mkdir(parents=True, exist_ok=True)
            self.params_file = self.run_path / "params.json"
            self.metrics_file = self.run_path / "metrics.json"
            self._params = self._load_json(self.params_file, {})
            self._metrics = self._load_json(self.metrics_file, {})
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"run_{timestamp}"
    
    def _load_json(self, path: Path, default: dict) -> dict:
        """Load JSON or return default."""
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return default
    
    def _save_json(self, path: Path, data: dict) -> None:
        """Save dict to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        if name not in self._metrics:
            self._metrics[name] = []
        entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        self._metrics[name].append(entry)
        if hasattr(self, "metrics_file"):
            self._save_json(self.metrics_file, self._metrics)
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter."""
        if not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        self._params[name] = value
        if hasattr(self, "params_file"):
            self._save_json(self.params_file, self._params)
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self._run_id
