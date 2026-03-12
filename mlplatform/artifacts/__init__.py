"""Standalone artifact storage — create ArtifactRegistry from config or explicit params.

Use this module to save/load artifacts without running the full framework.
Standalone mode only — explicit params.
"""

from mlplatform.artifacts.core import Artifact, ArtifactConfig, ArtifactRegistry, create_artifacts

__all__ = ["Artifact", "create_artifacts", "ArtifactConfig", "ArtifactRegistry"]
