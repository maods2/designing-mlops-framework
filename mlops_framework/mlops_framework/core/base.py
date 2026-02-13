"""Base classes for ML pipelines."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from mlops_framework.core.runtime import Runtime
from mlops_framework.artifacts.types import ArtifactType


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    
    Data Scientists must implement the core methods: load_data, train,
    evaluate, and predict. The framework handles the rest (tracking, artifacts, etc.).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.runtime = Runtime(config_path)
        self.config = self.runtime.get_config()
        self.tracking = self.runtime.get_tracking_backend()
        self.artifacts = self.runtime.get_artifact_manager()
        self.model: Optional[Any] = None
    
    @abstractmethod
    def load_data(self, *args, **kwargs) -> Any:
        """
        Load training or inference data.
        
        This method must be implemented by the Data Scientist.
        
        Returns:
            Loaded data (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def train(self, data: Any, *args, **kwargs) -> Any:
        """
        Train the model.
        
        This method must be implemented by the Data Scientist.
        
        Args:
            data: Training data (from load_data)
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, data: Any, *args, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model.
        
        This method must be implemented by the Data Scientist.
        
        Args:
            model: Trained model
            data: Evaluation data
            
        Returns:
            Dictionary of metric name -> metric value
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, data: Any, *args, **kwargs) -> Any:
        """
        Generate predictions.
        
        This method must be implemented by the Data Scientist.
        
        Args:
            model: Trained model
            data: Inference data
            
        Returns:
            Predictions (format depends on implementation)
        """
        pass


class TrainingPipeline(BaseModel):
    """
    Base class for training pipelines.
    
    .. deprecated::
        Use PreprocessStep, TrainStep with LocalRunner for step-based pipelines.
    
    Manages the complete training lifecycle: data loading → training →
    evaluation → artifact saving. Automatically logs parameters, metrics,
    and artifacts.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training pipeline."""
        warnings.warn(
            "TrainingPipeline is deprecated. Use PreprocessStep and TrainStep "
            "with LocalRunner for step-based pipelines.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config_path)
        self.trained_model: Optional[Any] = None
        self.metrics: Dict[str, float] = {}
    
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Args:
            *args, **kwargs: Arguments passed to load_data
            
        Returns:
            Dictionary with run_id, metrics, and model path
        """
        # Log pipeline start
        if self.tracking:
            self.tracking.set_tags({
                "pipeline_type": "training",
                "run_id": self.runtime.get_run_id()
            })
        
        # Step 1: Load data
        print("Loading training data...")
        train_data = self.load_data(*args, **kwargs)
        
        # Step 2: Train model
        print("Training model...")
        self.trained_model = self.train(train_data, *args, **kwargs)
        self.model = self.trained_model
        
        # Step 3: Evaluate model
        print("Evaluating model...")
        eval_data = self.load_eval_data(*args, **kwargs) if hasattr(self, 'load_eval_data') else train_data
        self.metrics = self.evaluate(self.trained_model, eval_data, *args, **kwargs)
        
        # Step 4: Log metrics
        if self.tracking:
            for metric_name, metric_value in self.metrics.items():
                self.tracking.log_metric(metric_name, metric_value)
        
        # Step 5: Save model and artifacts
        print("Saving artifacts...")
        model_path = self.artifacts.save(
            self.trained_model,
            "model",
            ArtifactType.MODEL
        )
        
        # Log model to tracking backend
        if self.tracking:
            self.tracking.log_model(
                self.trained_model,
                "model",
                metadata={"metrics": self.metrics}
            )
        
        # Save additional artifacts if provided
        self.save_artifacts(*args, **kwargs)
        
        print(f"Training complete! Run ID: {self.runtime.get_run_id()}")
        
        return {
            "run_id": self.runtime.get_run_id(),
            "metrics": self.metrics,
            "model_path": model_path
        }
    
    def save_artifacts(self, *args, **kwargs) -> None:
        """
        Save additional artifacts (overridable by subclasses).
        
        Subclasses can override this to save feature lists, scalers, encoders, etc.
        """
        pass
    
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter (convenience method)."""
        if self.tracking:
            self.tracking.log_param(name, value)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric (convenience method)."""
        if self.tracking:
            self.tracking.log_metric(name, value, step)


class InferencePipeline(BaseModel):
    """
    Base class for inference pipelines.
    
    .. deprecated::
        Use InferenceStep with LocalRunner for step-based pipelines.
    
    Supports both batch and online inference. Loads trained models from
    previous runs and generates predictions.
    """
    
    def __init__(self, config_path: Optional[str] = None, run_id: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            config_path: Optional path to configuration file
            run_id: Optional run ID to load model from (uses latest if None)
        """
        warnings.warn(
            "InferencePipeline is deprecated. Use InferenceStep with LocalRunner.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config_path)
        self.source_run_id = run_id
        self.loaded_model: Optional[Any] = None
    
    def load_model(self, run_id: Optional[str] = None, artifact_name: str = "model") -> Any:
        """
        Load a trained model from a previous run.
        
        Args:
            run_id: Run ID to load from (uses source_run_id if None)
            artifact_name: Name of the model artifact
            
        Returns:
            Loaded model
        """
        if run_id is None:
            run_id = self.source_run_id
        
        if run_id is None:
            raise ValueError("Must provide run_id to load model")
        
        # Create artifact manager for the source run
        from mlops_framework.artifacts.manager import ArtifactManager
        artifacts_config = self.config.get_artifacts_config()
        base_path = artifacts_config.get("base_path", "./artifacts")
        
        source_artifacts = ArtifactManager(
            base_path=base_path,
            run_id=run_id,
            tracking_backend=None
        )
        
        self.loaded_model = source_artifacts.load(artifact_name, ArtifactType.MODEL)
        self.model = self.loaded_model
        
        return self.loaded_model
    
    def run_batch(self, *args, **kwargs) -> Any:
        """
        Execute batch inference.
        
        Args:
            *args, **kwargs: Arguments passed to load_data
            
        Returns:
            Predictions
        """
        # Load model if not already loaded
        if self.loaded_model is None:
            if self.source_run_id:
                print(f"Loading model from run: {self.source_run_id}")
                self.load_model()
            else:
                raise ValueError("No model loaded. Call load_model() first or provide run_id in __init__")
        
        # Load inference data
        print("Loading inference data...")
        inference_data = self.load_data(*args, **kwargs)
        
        # Generate predictions
        print("Generating predictions...")
        predictions = self.predict(self.loaded_model, inference_data, *args, **kwargs)
        
        return predictions
    
    def run_online(self, input_data: Any, *args, **kwargs) -> Any:
        """
        Execute online inference (single prediction).
        
        Args:
            input_data: Single input for prediction
            *args, **kwargs: Additional arguments
            
        Returns:
            Single prediction
        """
        # Load model if not already loaded
        if self.loaded_model is None:
            if self.source_run_id:
                self.load_model()
            else:
                raise ValueError("No model loaded. Call load_model() first or provide run_id in __init__")
        
        # Generate prediction
        prediction = self.predict(self.loaded_model, input_data, *args, **kwargs)
        
        return prediction
