"""Dataproc Spark job runner - submits steps to Dataproc cluster."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlplatform.runners.base import JobRunner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class DataprocJobRunner(JobRunner):
    """Execute steps on a Dataproc Spark cluster.

    Submits job via: gcloud dataproc jobs submit pyspark
    Requires main.py as entry point and root.zip as --py-files.
    """

    def __init__(
        self,
        cluster: str,
        region: str | None = None,
        project: str | None = None,
        staging_bucket: str | None = None,
        main_script: str | Path | None = None,
        model_package: str = "example_model",
    ) -> None:
        self.cluster = cluster
        self.region = region
        self.project = project
        self.staging_bucket = staging_bucket
        self.main_script = main_script or self._default_main_script()
        self.model_package = model_package

    def _default_main_script(self) -> Path:
        import mlplatform.spark.main as spark_main
        return Path(spark_main.__file__).resolve()

    def execute(self, step: "Step", context: "ExecutionContext") -> Any:
        from mlplatform.spark.packager import build_root_zip

        project_root = Path(context.environment_metadata.get("project_root", ".")).resolve()
        monorepo_root = project_root.parent
        dist_dir = project_root / "dist"

        root_zip = build_root_zip(
            project_root=monorepo_root,
            model_package=project_root.name,
            output_dir=dist_dir,
        )

        config_path = dist_dir / "run_config.json"
        self._write_context_config(context, config_path)

        if self.staging_bucket:
            model_name = context.runtime_config.get("model_name", "default")
            version = context.runtime_config.get("version", "dev")
            gs_root = f"gs://{self.staging_bucket}/mlplatform/{model_name}/{version}"
            gs_zip = f"{gs_root}/root.zip"
            gs_config = f"{gs_root}/run_config.json"
            self._upload_to_gcs(root_zip, config_path, gs_zip, gs_config)
            packages_arg = gs_zip
            config_arg = gs_config
        else:
            packages_arg = str(root_zip)
            config_arg = str(config_path)

        cmd = [
            "gcloud", "dataproc", "jobs", "submit", "pyspark",
            str(self.main_script),
            f"--cluster={self.cluster}",
            f"--py-files={packages_arg}",
            "--",
            f"--config={config_arg}",
        ]
        if self.region:
            cmd.insert(-1, f"--region={self.region}")
        if self.project:
            cmd.insert(-1, f"--project={self.project}")

        subprocess.run(cmd, check=True)
        return None

    @staticmethod
    def _write_context_config(context: Any, config_path: Path) -> None:
        import json
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump({
                "runtime_config": context.runtime_config,
                "environment_metadata": context.environment_metadata,
            }, f, indent=2)

    @staticmethod
    def _upload_to_gcs(local_zip: Path, local_config: Path, gs_zip: str, gs_config: str) -> None:
        try:
            subprocess.run(["gsutil", "cp", str(local_zip), gs_zip], check=True)
            subprocess.run(["gsutil", "cp", str(local_config), gs_config], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"GCS upload failed: {e}") from e
