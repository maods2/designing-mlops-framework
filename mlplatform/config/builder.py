"""PipelineConfigBuilder — incremental, validated construction of PipelineConfig."""

from __future__ import annotations

from typing import Any, Self

from mlplatform.config.models import PipelineConfig


class PipelineConfigBuilder:
    """Builder for PipelineConfig with incremental validation.

    Example::

        config = (
            PipelineConfigBuilder()
            .identity(model_name="churn_model", feature="churn", version="v1.2")
            .infra(backend="gcs", base_bucket="ml-artifacts", project_id="my-project")
            .pipeline(pipeline_type="training", profile="cloud-train")
            .configs(["global", "train-prod"], config_dir="./config")
            .build()
        )
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def identity(
        self,
        *,
        model_name: str,
        feature: str,
        version: str = "dev",
    ) -> Self:
        """Set model identity fields."""
        self._data["model_name"] = model_name
        self._data["feature"] = feature
        self._data["version"] = version
        return self

    def infra(
        self,
        *,
        backend: str = "local",
        base_path: str = "./artifacts",
        base_bucket: str | None = None,
        project_id: str | None = None,
    ) -> Self:
        """Set infrastructure / storage fields."""
        self._data["backend"] = backend
        self._data["base_path"] = base_path
        if base_bucket is not None:
            self._data["base_bucket"] = base_bucket
        if project_id is not None:
            self._data["project_id"] = project_id
        return self

    def pipeline(
        self,
        *,
        pipeline_type: str,
        profile: str = "local",
        platform: str = "VertexAI",
        module: str = "",
    ) -> Self:
        """Set pipeline execution fields."""
        self._data["pipeline_type"] = pipeline_type
        self._data["profile"] = profile
        self._data["platform"] = platform
        if module:
            self._data["module"] = module
        return self

    def configs(
        self,
        profile_names: list[str],
        config_dir: str = "./config",
    ) -> Self:
        """Set config profile names and directory for YAML merging."""
        self._data["config_list"] = profile_names
        self._data["config_dir"] = config_dir
        return self

    def user_config(self, config: dict[str, Any]) -> Self:
        """Set merged user config dict directly."""
        self._data["user_config"] = config
        return self

    def metadata(
        self,
        *,
        commit_hash: str | None = None,
        log_level: str = "INFO",
    ) -> Self:
        """Set metadata fields."""
        if commit_hash is not None:
            self._data["commit_hash"] = commit_hash
        self._data["log_level"] = log_level
        return self

    def build(self) -> PipelineConfig:
        """Validate and return a frozen PipelineConfig.

        Raises:
            ValueError: If required fields are missing or invalid combinations detected.
        """
        if "model_name" not in self._data:
            raise ValueError("PipelineConfigBuilder requires .identity(model_name=...)")
        if "feature" not in self._data:
            raise ValueError("PipelineConfigBuilder requires .identity(feature=...)")
        if "pipeline_type" not in self._data:
            raise ValueError("PipelineConfigBuilder requires .pipeline(pipeline_type=...)")

        backend = self._data.get("backend", "local")
        base_bucket = self._data.get("base_bucket")
        if backend == "gcs" and not base_bucket:
            uc = self._data.get("user_config", {})
            if not uc.get("base_bucket"):
                raise ValueError("backend='gcs' requires base_bucket")

        return PipelineConfig.from_dict(self._data)
