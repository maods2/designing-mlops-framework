"""Local Spark job runner - run steps locally with optional spark-submit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlplatform.runners.base import JobRunner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step


class LocalSparkJobRunner(JobRunner):
    """Execute steps locally. Supports direct in-process or local spark-submit.

    Modes:
    1. direct=True (default): Run step in-process (no Spark, fast)
    2. spark_submit=True: Run spark-submit with main.py for local testing of cloud flow
       - Builds root.zip and run_config.json, then submits via spark-submit
    """

    def __init__(
        self,
        direct: bool = True,
        spark_submit: bool = False,
        main_script: str | Path | None = None,
        model_package: str = "example_model",
    ) -> None:
        self.direct = direct
        self.spark_submit = spark_submit
        self.main_script = main_script or self._default_main_script()
        self.model_package = model_package

    def _default_main_script(self) -> Path:
        import mlplatform.spark.main as spark_main
        return Path(spark_main.__file__).resolve()

    def execute(self, step: "Step", context: "ExecutionContext") -> Any:
        if self.direct and not self.spark_submit:
            return step.run(context)

        from mlplatform.spark.packager import build_root_zip
        project_root = Path(context.environment_metadata.get("project_root", ".")).resolve()
        monorepo_root = project_root.parent
        dist_dir = project_root / "dist"

        root_zip = build_root_zip(
            project_root=monorepo_root,
            model_package=project_root.name,
            output_dir=dist_dir,
        )

        if self.spark_submit:
            return self._run_spark_submit(root_zip, context)
        return self._run_python_main(root_zip, context)

    def _run_python_main(self, root_zip: Path, context: Any) -> Any:
        config_path = root_zip.parent / "run_config.json"
        self._write_context_config(context, config_path)
        cmd = [
            sys.executable,
            str(self.main_script),
            f"--config={config_path}",
            f"--packages={root_zip}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(config_path.parent.parent))
        if result.returncode != 0:
            raise RuntimeError(f"main.py failed: {result.stderr}")
        return result.stdout

    def _run_spark_submit(self, root_zip: Path, context: Any) -> Any:
        config_path = root_zip.parent / "run_config.json"
        self._write_context_config(context, config_path)
        cmd = [
            "spark-submit",
            "--master", "local[*]",
            "--py-files", str(root_zip),
            str(self.main_script),
            f"--config={config_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"spark-submit failed: {result.stderr}")
        return result.stdout

    @staticmethod
    def _write_context_config(context: Any, config_path: Path) -> None:
        import json
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump({
                "runtime_config": context.runtime_config,
                "environment_metadata": context.environment_metadata,
            }, f, indent=2)
