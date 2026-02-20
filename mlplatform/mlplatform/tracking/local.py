"""Local JSON experiment tracker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform.tracking.base import ExperimentTracker


class LocalJsonTracker(ExperimentTracker):
    """Track experiments by appending to a local JSON file."""

    def __init__(self, base_path: str = "./artifacts", run_id: str | None = None) -> None:
        self.base_path = Path(base_path)
        self.run_id = run_id or "default"
        self._run_data: dict[str, Any] = {"params": {}, "metrics": {}, "artifacts": []}

    def log_params(self, params: dict[str, Any]) -> None:
        self._run_data["params"].update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._run_data["metrics"].update(metrics)

    def log_artifact(self, path: str, artifact: Any) -> None:
        self._run_data["artifacts"].append({"path": path, "artifact": str(artifact)})

    def save(self, output_path: str | None = None) -> None:
        """Persist run data to JSON file."""
        path = Path(output_path) if output_path else self.base_path / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._run_data, f, indent=2)
