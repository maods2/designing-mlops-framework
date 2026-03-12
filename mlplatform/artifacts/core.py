"""Standalone artifact storage — create ArtifactRegistry from explicit params.

Use this module to save/load artifacts without running the full framework.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mlplatform.artifacts.registry import ArtifactRegistry

if TYPE_CHECKING:
    from mlplatform.storage.base import Storage

_logger = logging.getLogger(__name__)


@dataclass
class ArtifactConfig:
    """Internal config for artifact storage. Built from standalone params."""

    backend: Literal["local", "gcs"] | None
    base_path: str
    project: str | None
    feature_name: str
    model_name: str
    version: str


def create_artifacts(
    *,
    backend: Literal["local", "gcs"] | None = None,
    bucket: str | None = None,
    base_bucket: str | None = None,
    prefix: str = "",
    base_path: str = "./artifacts",
    project: str | None = None,
    feature_name: str | None = None,
    feature: str | None = None,
    model_name: str,
    version: str = "dev",
) -> ArtifactRegistry:
    """Create an ArtifactRegistry for saving/loading model artifacts.

    Example::

        artifacts = create_artifacts(
            backend="gcs",
            bucket="my-bucket",
            prefix="models",
            project="my-gcp-project",
            feature_name="feature",
            model_name="model",
            version="v1",
        )

    Returns:
        ArtifactRegistry with save(name, obj) and load(name, ...) methods.
    """
    resolved_feature = feature_name or feature
    if resolved_feature is None:
        raise ValueError("create_artifacts requires 'feature_name' or 'feature'.")

    if backend is None:
        _logger.warning(
            "Storage backend not activated. Set backend='local' or backend='gcs' "
            "to persist artifacts. save/load will no-op or raise until then."
        )

    cfg = _config_from_standalone(
        backend=backend,
        bucket=bucket,
        prefix=prefix,
        base_path=base_path,
        base_bucket=base_bucket,
        project=project,
        feature_name=resolved_feature,
        model_name=model_name,
        version=version,
    )

    storage = _create_storage(cfg)
    return ArtifactRegistry(
        storage=storage,
        feature_name=cfg.feature_name,
        model_name=cfg.model_name,
        version=cfg.version,
    )


def _config_from_standalone(
    *,
    backend: Literal["local", "gcs"] | None,
    bucket: str | None,
    prefix: str,
    base_path: str,
    base_bucket: str | None = None,
    project: str | None,
    feature_name: str,
    model_name: str,
    version: str,
) -> ArtifactConfig:
    """Build ArtifactConfig from standalone params."""
    if backend is None:
        return ArtifactConfig(
            backend=None,
            base_path=base_path or "./artifacts",
            project=project,
            feature_name=feature_name,
            model_name=model_name,
            version=version,
        )
    if backend == "gcs":
        resolved_bucket = bucket or base_bucket
        if not resolved_bucket:
            raise ValueError("GCS backend requires 'bucket' or 'base_bucket'.")
        prefix_part = prefix.rstrip("/") if prefix else ""
        resolved_base_path = f"gs://{resolved_bucket}/{prefix_part}" if prefix_part else f"gs://{resolved_bucket}"
        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    else:
        resolved_base_path = base_bucket or base_path
        resolved_project = None

    return ArtifactConfig(
        backend=backend,
        base_path=resolved_base_path,
        project=resolved_project,
        feature_name=feature_name,
        model_name=model_name,
        version=version,
    )


def _create_storage(cfg: ArtifactConfig) -> Storage:
    """Create Storage backend from ArtifactConfig."""
    if cfg.backend is None:
        from mlplatform.storage.none import NoneStorage
        return NoneStorage()
    if cfg.backend == "local":
        from mlplatform.storage.local import LocalFileSystem
        return LocalFileSystem(cfg.base_path)
    else:
        from mlplatform.storage.gcs import GCSStorage
        return GCSStorage(cfg.base_path, project=cfg.project)


def Artifact(
    *,
    project: str | None = None,
    project_id: str | None = None,
    model_name: str,
    feature: str | None = None,
    feature_name: str | None = None,
    version: str = "dev",
    base_bucket: str | None = None,
    bucket: str | None = None,
    base_path: str = "./artifacts",
    backend: Literal["local", "gcs"] | None = None,
    prefix: str = "",
) -> ArtifactRegistry:
    """Convenience constructor for ArtifactRegistry.

    Accepts both user-friendly param names (project_id, feature, base_bucket)
    and canonical names (project, feature_name, bucket).

    - ``base_bucket``: for GCS = bucket name; for local = base directory path
    - ``feature`` / ``feature_name``: feature domain (e.g. "churn")

    Example::

        artifact = Artifact(
            project_id="my-gcp-project",
            model_name="churn_model",
            feature="churn",
            version="v1",
            base_bucket="my-bucket",
            backend="gcs",
        )
        artifact.save("model.pkl", model)
    """
    resolved_project = project or project_id
    resolved_feature = feature_name or feature
    if resolved_feature is None:
        raise ValueError("Artifact requires 'feature' or 'feature_name'.")

    if backend is None:
        _logger.warning(
            "Storage backend not activated. Set backend='local' or backend='gcs' "
            "to persist artifacts. save/load will no-op or raise until then."
        )

    if backend == "gcs":
        resolved_bucket = bucket or base_bucket
        resolved_base_path = base_path
    else:
        resolved_bucket = bucket
        resolved_base_path = base_bucket or base_path

    return create_artifacts(
        backend=backend,
        bucket=resolved_bucket,
        base_bucket=base_bucket,
        prefix=prefix,
        base_path=resolved_base_path,
        project=resolved_project,
        feature_name=resolved_feature,
        model_name=model_name,
        version=version,
    )
