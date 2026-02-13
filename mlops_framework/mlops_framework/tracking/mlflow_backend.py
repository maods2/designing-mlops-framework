"""MLflow-based tracking backend (optional)."""

from typing import Any, Dict, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from mlops_framework.tracking.interface import TrackingBackend
from mlops_framework.tracking.local import LocalTrackingBackend


class MLflowTrackingBackend(TrackingBackend):
    """
    MLflow-based tracking backend.
    
    Wraps MLflow tracking API. Falls back to LocalTrackingBackend
    if MLflow is not available.
    """
    
    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "default", 
                 run_id: Optional[str] = None, fallback_to_local: bool = True):
        """
        Initialize the MLflow tracking backend.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
            run_id: Optional run ID (MLflow will create one if None)
            fallback_to_local: If True, fall back to local backend if MLflow unavailable
        """
        if not MLFLOW_AVAILABLE:
            if fallback_to_local:
                # Fall back to local backend
                self._fallback = LocalTrackingBackend(run_id=run_id)
                self._use_fallback = True
                return
            else:
                raise ImportError(
                    "MLflow is not installed. Install it with: pip install mlflow"
                )
        
        self._use_fallback = False
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run()
        
        self.run_id = mlflow.active_run().info.run_id
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter."""
        if self._use_fallback:
            return self._fallback.log_param(name, value)
        
        # MLflow requires string values for params
        mlflow.log_param(name, str(value))
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if self._use_fallback:
            return self._fallback.log_metric(name, value, step)
        
        mlflow.log_metric(name, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact."""
        if self._use_fallback:
            return self._fallback.log_artifact(local_path, artifact_path)
        
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model: Any, artifact_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a trained model."""
        if self._use_fallback:
            return self._fallback.log_model(model, artifact_path, metadata)
        
        # Try to use MLflow's model logging if it's a supported framework
        # Otherwise, use generic artifact logging
        try:
            # For scikit-learn models
            if hasattr(model, 'predict') and hasattr(model, 'fit'):
                mlflow.sklearn.log_model(model, artifact_path)
            else:
                # Generic model logging
                import pickle
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                    pickle.dump(model, f)
                    temp_path = f.name
                
                try:
                    mlflow.log_artifact(temp_path, artifact_path)
                finally:
                    os.unlink(temp_path)
        except Exception:
            # Fallback to generic artifact logging
            import pickle
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                pickle.dump(model, f)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, artifact_path)
            finally:
                os.unlink(temp_path)
        
        # Log metadata if provided
        if metadata:
            mlflow.log_dict(metadata, f"{artifact_path}.metadata.json")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current run."""
        if self._use_fallback:
            return self._fallback.set_tags(tags)
        
        mlflow.set_tags(tags)
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        if self._use_fallback:
            return self._fallback.get_run_id()
        
        return mlflow.active_run().info.run_id
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end MLflow run."""
        if not self._use_fallback:
            mlflow.end_run()
