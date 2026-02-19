"""Dataproc Spark runner - submits steps to Dataproc cluster."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlplatform.config.schema import RunConfig
from mlplatform.runners.base import Runner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step
from mlplatform.spark.config_serializer import write_run_config
from mlplatform.spark.packager import build_root_zip


class DataprocSparkRunner(Runner):
    """
    Execute steps on a Dataproc Spark cluster.

    Submits job via: gcloud dataproc jobs submit pyspark
    Format: mainscript=main.py, packages=root.zip
    """

    def __init__(
        self,
        cluster: str,
        region: str | None = None,
        project: str | None = None,
        staging_bucket: str | None = None,
        main_script: str | Path | None = None,
        model_package: str = "template_model",
    ) -> None:
        self.cluster = cluster
        self.region = region
        self.project = project
        self.staging_bucket = staging_bucket
        self.main_script = main_script or self._default_main_script()
        self.model_package = model_package

    def _default_main_script(self) -> Path:
        """Path to framework's main.py."""
        import mlplatform.spark.main as spark_main

        return Path(spark_main.__file__).resolve()

    def run(self, step: "Step", context: "ExecutionContext", **kwargs: Any) -> Any:
        """
        Submit step to Dataproc. Does not wait for result or return in-process.
        For async execution; orchestrator polls job status.
        """
        run_config = context.run_config
        # project_root = model project dir (e.g. template_model/)
        project_root = Path(run_config.env_config.extra.get("project_root", ".")).resolve()
        monorepo_root = project_root.parent

        # Build root.zip under model project
        dist_dir = project_root / "dist"
        root_zip = build_root_zip(
            project_root=monorepo_root,
            model_package=project_root.name,
            output_dir=dist_dir,
        )

        # Write config to temp/local. base_path injected (orchestrator provides in prod)
        dist_dir = Path(project_root) / "dist"
        config_path = dist_dir / "run_config.json"
        base_path = str(Path(project_root) / "artifacts")
        write_run_config(run_config, config_path, base_path=base_path)

        # Upload to GCS if staging_bucket set
        if self.staging_bucket:
            gs_root = f"gs://{self.staging_bucket}/mlplatform/{run_config.model_name}/{run_config.version}"
            gs_zip = f"{gs_root}/root.zip"
            gs_config = f"{gs_root}/run_config.json"
            self._upload_to_gcs(root_zip, config_path, gs_zip, gs_config)
            packages_arg = gs_zip
            config_arg = gs_config
        else:
            packages_arg = str(root_zip)
            config_arg = str(config_path)

        cmd = [
            "gcloud",
            "dataproc",
            "jobs",
            "submit",
            "pyspark",
            str(self.main_script),
            f"--cluster={self.cluster}",
            f"--py-files={packages_arg}",
            "--",
            f"--config={config_arg}",
            f"--step-name={run_config.step.name}",
        ]
        if self.region:
            cmd.insert(-1, f"--region={self.region}")
        if self.project:
            cmd.insert(-1, f"--project={self.project}")

        subprocess.run(cmd, check=True)
        return None

    def _upload_to_gcs(self, local_zip: Path, local_config: Path, gs_zip: str, gs_config: str) -> None:
        """Upload root.zip and config to GCS."""
        try:
            subprocess.run(["gsutil", "cp", str(local_zip), gs_zip], check=True)
            subprocess.run(["gsutil", "cp", str(local_config), gs_config], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"GCS upload failed: {e}") from e
