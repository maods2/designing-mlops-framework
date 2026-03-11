"""Standalone artifact storage — create ArtifactRegistry from config or explicit params.

Use this module to save/load artifacts without running the full framework.
Supports both config-driven (workflow + model_cfg) and standalone (explicit params) modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mlplatform.core.artifact_registry import ArtifactRegistry

if TYPE_CHECKING:
    from mlplatform.config.models import ModelConfig, WorkflowConfig
    from mlplatform.profiles.registry import Profile
    from mlplatform.storage.base import Storage


@dataclass
class ArtifactConfig:
    """Internal config for artifact storage. Built from workflow config or explicit params."""

    backend: Literal["local", "gcs"]
    base_path: str
    project: str | None
    feature_name: str
    model_name: str
    version: str


def create_artifacts(
    *,
    # Config-driven (framework) params
    workflow: WorkflowConfig | None = None,
    model_cfg: ModelConfig | None = None,
    version: str | None = None,
    profile: Profile | str | None = None,
    project: str | None = None,
    base_path_override: str | None = None,
    # Standalone params
    backend: Literal["local", "gcs"] | None = None,
    bucket: str | None = None,
    prefix: str = "",
    base_path: str = "./artifacts",
    feature_name: str | None = None,
    model_name: str | None = None,
) -> ArtifactRegistry:
    """Create an ArtifactRegistry for saving/loading model artifacts.

    Two usage modes:

    **1. From framework config** (orchestrator / workflow):

        artifacts = create_artifacts(
            workflow=workflow,
            model_cfg=model_cfg,
            version=version,
            profile=profile,
            project=project,
        )

    **2. Standalone** (scripts, notebooks — explicit params):

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
    if workflow is not None and model_cfg is not None and version is not None:
        cfg = _config_from_workflow(
            workflow=workflow,
            model_cfg=model_cfg,
            version=version,
            profile=profile,
            project=project,
            base_path_override=base_path_override,
        )
    elif backend is not None and feature_name is not None and model_name is not None:
        cfg = _config_from_standalone(
            backend=backend,
            bucket=bucket,
            prefix=prefix,
            base_path=base_path,
            project=project,
            feature_name=feature_name,
            model_name=model_name,
            version=version or "dev",
        )
    else:
        raise ValueError(
            "create_artifacts requires either (workflow, model_cfg, version) "
            "or (backend, feature_name, model_name, version)."
        )

    storage = _create_storage(cfg)
    return ArtifactRegistry(
        storage=storage,
        feature_name=cfg.feature_name,
        model_name=cfg.model_name,
        version=cfg.version,
    )


def _config_from_workflow(
    *,
    workflow: WorkflowConfig,
    model_cfg: ModelConfig,
    version: str,
    profile: Profile | str | None,
    project: str | None,
    base_path_override: str | None,
) -> ArtifactConfig:
    """Build ArtifactConfig from workflow + model config."""
    from mlplatform.profiles.registry import get_profile

    opt = model_cfg.optional_configs
    prof: Profile | None = None
    if profile is not None:
        prof = get_profile(profile) if isinstance(profile, str) else profile

    # Resolve project: override > profile.extra > config > env
    resolved_project = (
        project
        or (prof.extra.get("gcp_project") if prof else None)
        or opt.get("gcp_project")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    # Determine backend from profile name (cloud profiles use GCS)
    profile_name = prof.name if prof else (str(profile) if profile else "local")
    is_gcs = profile_name in ("cloud-batch", "cloud-online", "cloud-train")

    if base_path_override:
        base_path = base_path_override
    elif is_gcs:
        bucket = opt.get("bucket")
        if not bucket:
            raise ValueError(
                "GCS profile requires 'bucket' in config (optional_configs or config profile YAML)."
            )
        artifact_prefix = opt.get("artifact_prefix", "models")
        prefix = f"{artifact_prefix}".rstrip("/")
        base_path = f"gs://{bucket}/{prefix}" if prefix else f"gs://{bucket}"
    else:
        base_path = opt.get("base_path", "./artifacts")

    return ArtifactConfig(
        backend="gcs" if is_gcs else "local",
        base_path=base_path,
        project=resolved_project,
        feature_name=workflow.feature_name,
        model_name=model_cfg.model_name,
        version=version,
    )


def _config_from_standalone(
    *,
    backend: Literal["local", "gcs"],
    bucket: str | None,
    prefix: str,
    base_path: str,
    project: str | None,
    feature_name: str,
    model_name: str,
    version: str,
) -> ArtifactConfig:
    """Build ArtifactConfig from standalone params."""
    if backend == "gcs":
        if not bucket:
            raise ValueError("GCS backend requires 'bucket'.")
        prefix_part = prefix.rstrip("/") if prefix else ""
        resolved_base_path = f"gs://{bucket}/{prefix_part}" if prefix_part else f"gs://{bucket}"
        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    else:
        resolved_base_path = base_path
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
    if cfg.backend == "local":
        from mlplatform.storage.local import LocalFileSystem
        return LocalFileSystem(cfg.base_path)
    else:
        from mlplatform.storage.gcs import GCSStorage
        return GCSStorage(cfg.base_path, project=cfg.project)
