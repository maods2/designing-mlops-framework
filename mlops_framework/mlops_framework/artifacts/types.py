"""Artifact type definitions."""

from enum import Enum


class ArtifactType(Enum):
    """Semantic types for artifacts."""
    
    MODEL = "model"
    FEATURES = "features"
    SCALER = "scaler"
    ENCODER = "encoder"
    PLOT = "plot"
    METADATA = "metadata"
    OTHER = "other"
