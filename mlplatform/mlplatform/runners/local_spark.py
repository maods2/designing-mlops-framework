"""Local Spark runner - run steps locally without main.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlplatform.runners.base import Runner

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.steps import Step
from mlplatform.spark.config_serializer import write_run_config
from mlplatform.spark.main import run_spark_step
from mlplatform.spark.packager import build_root_zip


class LocalSparkRunner(Runner):
    """
    Execute steps locally in-process. Does NOT use main.py.

    main.py is an auxiliary job used ONLY during cloud job submission
    (spark-submit). For local execution we run the step directly.

    Modes:
    1. direct=True (default): Run step in-process via run_spark_step() (no Spark, fast)
    2. spark_submit=True: Run spark-submit with main.py for local testing of cloud flow
    """

    def __init__(
        self,
        direct: bool = True,
        spark_submit: bool = False,
        main_script: str | Path | None = None,
        model_package: str = "template_model",
    ) -> None:
        self.direct = direct
        self.spark_submit = spark_submit
        self.main_script = main_script or self._default_main_script()
        self.model_package = model_package

    def _default_main_script(self) -> Path:
        """Path to framework's main.py (used only for spark_submit cloud simulation)."""
        import mlplatform.spark.main as spark_main

        return Path(spark_main.__file__).resolve()

    def run(self, step: "Step", context: "ExecutionContext", **kwargs: Any) -> Any:
        run_config = context.run_config
        # project_root = model project dir (e.g. template_model/)
        project_root = Path(run_config.env_config.extra.get("project_root", ".")).resolve()
        if not project_root.exists():
            project_root = Path.cwd()
        # For packaging: monorepo root is parent of model project
        monorepo_root = project_root.parent

        if self.direct and not self.spark_submit:
            # Local path: run in-process, no main.py, no packaging
            from mlplatform.spark.config_serializer import run_config_to_dict
            import tempfile
            import json

            base_path = str(project_root / "artifacts")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(run_config_to_dict(run_config, base_path=base_path), f, indent=2)
                config_path = f.name
            try:
                return run_spark_step(config_path, run_config.step.name, **kwargs)
            finally:
                Path(config_path).unlink(missing_ok=True)

        # Optional: spark-submit with main.py for local testing of cloud flow
        dist_dir = project_root / "dist"
        base_path = str(project_root / "artifacts")
        root_zip = build_root_zip(
            project_root=monorepo_root,
            model_package=project_root.name,
            output_dir=dist_dir,
        )
        config_path = dist_dir / "run_config.json"
        write_run_config(run_config, config_path, base_path=base_path)

        if self.spark_submit:
            return self._run_spark_submit(root_zip, config_path, run_config.step.name, **kwargs)
        return self._run_python_main(root_zip, config_path, run_config.step.name, **kwargs)

    def _run_python_main(
        self,
        root_zip: Path,
        config_path: Path,
        step_name: str,
        **kwargs: Any,
    ) -> Any:
        """Run main.py via Python with --packages root.zip (full format simulation)."""
        cmd = [
            sys.executable,
            str(self.main_script),
            f"--config={config_path}",
            f"--packages={root_zip}",
            f"--step-name={step_name}",
        ]
        if kwargs.get("input_path"):
            cmd.append(f"--input-path={kwargs['input_path']}")
        if kwargs.get("output_path"):
            cmd.append(f"--output-path={kwargs['output_path']}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(config_path.parent.parent))
        if result.returncode != 0:
            raise RuntimeError(f"main.py failed: {result.stderr}")
        return result.stdout

    def _run_spark_submit(
        self,
        root_zip: Path,
        config_path: Path,
        step_name: str,
        **kwargs: Any,
    ) -> Any:
        """Run via spark-submit --master local[*] for local Spark testing."""
        cmd = [
            "spark-submit",
            "--master", "local[*]",
            "--py-files", str(root_zip),
            str(self.main_script),
            f"--config={config_path}",
            f"--step-name={step_name}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"spark-submit failed: {result.stderr}")
        return result.stdout
