"""ArtifactPathBuilder - standardizes artifact path structure for local and GCS.

Provides a unified path layout for model artifacts and metrics, compatible with
both local filesystem and Google Cloud Storage. Used by ArtifactRegistry,
tracking implementations, and config serialization.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional


@dataclass
class ArtifactPaths:
    """Output of ArtifactPathBuilder.build_artifact_paths."""

    artifact_bucket: str
    artifact_base_path: str
    artifact_path: str
    model_artifact_dir: str
    metrics_artifact_dir: str
    storage_base_path: str  # Full path for Storage (artifact_path + model/)
    metrics_path: str  # Full path to metrics.json
    full_model_name: str
    model_train_version: str
    run_id: str


class ArtifactPathBuilder:
    """
    Utility class to build standardized artifact paths from configuration.

    Handles environment-specific paths for both local and GCS storage.
    Path structure: {bucket}/{namespace}/{feature}/{model_train_version}/model|metrics/
    """

    def __init__(
        self,
        env: str = "local",
        artifact_bucket: Optional[str] = None,
        artifact_namespace: str = "artifacts",
    ) -> None:
        """
        Initialize the builder with target environment and paths.

        Args:
            env: Target environment (local, dev, qa, prod).
            artifact_bucket: Base storage path. For GCS: gs://bucket. For local: ./artifacts.
                If None, defaults to ./artifacts for local env, gs://aim-pae-{env}-source-code-bucket for cloud.
            artifact_namespace: Namespace under bucket (e.g. aim-deo-mlops-artifacts or artifacts).
        """
        self.env = env
        self.artifact_namespace = artifact_namespace.rstrip("/")
        if artifact_bucket is not None:
            self.artifact_bucket = artifact_bucket.rstrip("/")
        elif env == "local":
            self.artifact_bucket = "./artifacts"
        else:
            self.artifact_bucket = f"gs://aim-pae-{env}-source-code-bucket"

    def get_version(self) -> str:
        """
        Generate a dynamic version timestamp.

        Returns:
            Version string in format YYYY-MM-DD-HH-MM
        """
        return datetime.now().strftime("%Y-%m-%d-%H-%M")

    def generate_run_id(self) -> str:
        """Generate a short alphanumeric run identifier."""
        return re.sub(r"[^a-zA-Z0-9]", "", str(uuid.uuid4()))[2:10]

    def build_artifact_paths(
        self,
        feature_name: str,
        model_name: str,
        version: str,
        pipeline_type: Literal["training", "prediction"] = "training",
        version_override: Optional[str] = None,
        run_id_override: Optional[str] = None,
    ) -> ArtifactPaths:
        """
        Build standardized artifact paths for a model run.

        Args:
            feature_name: Feature name for the model.
            model_name: Model name identifier.
            version: Version string (uses get_version() if empty).
            pipeline_type: 'training' or 'prediction'.
            version_override: Optional explicit version (skips get_version).
            run_id_override: Optional explicit run_id.

        Returns:
            ArtifactPaths with all path components.
        """
        version_str = version or version_override or self.get_version()
        run_id = run_id_override or self.generate_run_id()

        suffix = "train" if pipeline_type == "training" else "predict"
        full_model_name = f"{feature_name}_{model_name}_{suffix}"
        model_train_version = f"{full_model_name}_{version_str}"

        # Build paths (all relative to artifact_bucket for GCS, or as full local path)
        artifact_base_path = f"{self.artifact_bucket}/{self.artifact_namespace}/{feature_name}"
        artifact_path = f"{artifact_base_path}/{model_train_version}/"
        model_artifact_dir = f"{model_train_version}/model"
        metrics_artifact_dir = f"{model_train_version}/metrics"

        storage_base_path = f"{artifact_path}model"
        metrics_path = f"{artifact_path}metrics/metrics.json"

        return ArtifactPaths(
            artifact_bucket=self.artifact_bucket,
            artifact_base_path=artifact_base_path,
            artifact_path=artifact_path,
            model_artifact_dir=model_artifact_dir,
            metrics_artifact_dir=metrics_artifact_dir,
            storage_base_path=storage_base_path,
            metrics_path=metrics_path,
            full_model_name=full_model_name,
            model_train_version=model_train_version,
            run_id=run_id,
        )

    def build_artifact_paths_dict(
        self,
        feature_name: str,
        model_name: str,
        version: str,
        pipeline_type: Literal["training", "prediction"] = "training",
        version_override: Optional[str] = None,
        run_id_override: Optional[str] = None,
    ) -> dict[str, Any]:
        """Same as build_artifact_paths but returns a dict for serialization."""
        paths = self.build_artifact_paths(
            feature_name=feature_name,
            model_name=model_name,
            version=version,
            pipeline_type=pipeline_type,
            version_override=version_override,
            run_id_override=run_id_override,
        )
        return {
            "artifact_bucket": paths.artifact_bucket,
            "artifact_base_path": paths.artifact_base_path,
            "artifact_path": paths.artifact_path,
            "model_artifact_dir": paths.model_artifact_dir,
            "metrics_artifact_dir": paths.metrics_artifact_dir,
            "storage_base_path": paths.storage_base_path,
            "metrics_path": paths.metrics_path,
            "full_model_name": paths.full_model_name,
            "model_train_version": paths.model_train_version,
            "run_id": paths.run_id,
        }
