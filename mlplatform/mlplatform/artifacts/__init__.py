"""Standalone artifact storage — create ArtifactRegistry from config or explicit params.

Use this module to save/load artifacts without running the full framework.
Supports both config-driven (workflow + model_cfg) and standalone (explicit params) modes.
"""

from mlplatform.artifacts.core import ArtifactConfig, ArtifactRegistry, create_artifacts

__all__ = ["create_artifacts", "ArtifactConfig", "ArtifactRegistry"]
