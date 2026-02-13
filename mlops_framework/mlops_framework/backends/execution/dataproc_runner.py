"""Dataproc execution runner (placeholder for Spark/large-scale jobs)."""

from typing import Any, Dict, Optional, Type

from mlops_framework.backends.execution.base import BaseRunner
from mlops_framework.core.step import BaseStep


class DataprocRunner(BaseRunner):
    """
    Placeholder for Dataproc-based execution.
    
    Use for Spark jobs or large-scale preprocessing/training.
    To implement: submit jobs to Dataproc cluster.
    """
    
    def __init__(self, cluster_name: Optional[str] = None):
        """
        Initialize Dataproc runner (placeholder).
        
        Args:
            cluster_name: Optional Dataproc cluster name
        """
        self.cluster_name = cluster_name
    
    def run_step(
        self,
        step_class: Type[BaseStep],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Placeholder - not implemented."""
        raise NotImplementedError(
            "DataprocRunner is a placeholder. Implement job submission to Dataproc."
        )
