"""Run context schema for orchestrator-injected runtime parameters."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    """
    Schema for runtime parameters.

    Filled by orchestrator (Airflow, Vertex) in cloud; derived from defaults locally.
    Used to configure storage, tracking, and paths.
    """

    base_path: str = Field(default="./artifacts", description="Base path for artifacts")
    run_id: str = Field(default="run_default", description="Unique run identifier")
    model_name: Optional[str] = Field(default=None, description="Model identifier")
    model_id: Optional[str] = Field(default=None, description="Model version/registry ID")
    artifacts_path: Optional[str] = Field(default=None, description="Override path for artifacts")
    tracking_enabled: bool = Field(default=False, description="Whether tracking is enabled")
    tracking_backend: Literal["noop", "local", "vertex"] = Field(
        default="noop",
        description="Tracking backend: noop, local, or vertex",
    )
    experiment_name: Optional[str] = Field(default=None, description="Experiment name for Vertex")
    vertex_project: Optional[str] = Field(default=None, description="GCP project for Vertex AI")
    vertex_location: Optional[str] = Field(default=None, description="GCP region for Vertex AI")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunContext":
        """Build RunContext from dict (e.g. JSON from orchestrator). Backward-compat alias."""
        return cls.model_validate(d)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for XCom/env passing. Backward-compat alias."""
        return self.model_dump()
