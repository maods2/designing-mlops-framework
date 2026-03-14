"""Serialize PipelineConfig for Spark/Dataproc main.py consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform.config.models import PipelineConfig


def pipeline_config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Serialize a PipelineConfig to a JSON-serializable dict."""
    return config.model_dump()


def write_pipeline_config(
    config: PipelineConfig,
    path: str | Path,
) -> Path:
    """Write PipelineConfig to JSON file for cloud submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pipeline_config_to_dict(config), f, indent=2)
    return path
