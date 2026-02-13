"""CLI entry point for MLOps framework."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import yaml


def _parse_run_context(value: str):
    """Parse RunContext from JSON string or path to JSON file."""
    from mlops_framework.core.run_context import RunContext
    path = Path(value)
    if path.exists():
        with open(path, "r") as f:
            d = json.load(f)
    else:
        d = json.loads(value)
    return RunContext.model_validate(d)


def _add_project_root_to_path(project_root: Path) -> None:
    """Add project root to sys.path so step classes (e.g. steps.preprocess.ChurnPreprocess) resolve."""
    root_str = str(project_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_config(config_path: str) -> dict:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _merge_step_config(full_config: dict, step_id: str, env: str) -> dict:
    """Merge steps[step_id] with environments[env][step_id] for step-specific config."""
    steps_cfg = full_config.get("steps", {})
    step_base = steps_cfg.get(step_id, {}).copy()
    env_overrides = full_config.get("environments", {}).get(env, {}).get(step_id, {})
    # Deep merge: base + env overrides
    for k, v in env_overrides.items():
        if isinstance(v, dict) and k in step_base and isinstance(step_base[k], dict):
            step_base[k] = {**step_base[k], **v}
        else:
            step_base[k] = v
    return step_base


def _resolve_step_class(class_path: str):
    """Import and return step class from dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single step locally or in cloud mode."""
    project_root = Path(args.project_root or os.getcwd())
    _add_project_root_to_path(project_root)

    # Load pipeline YAML (pipeline/pipeline.yaml or --pipeline)
    pipeline_path = project_root / "pipeline" / "pipeline.yaml"
    if not pipeline_path.exists():
        pipeline_path = project_root / args.pipeline
    if not pipeline_path.exists():
        print(f"Error: Pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 1

    with open(pipeline_path, "r") as f:
        pipeline_def = yaml.safe_load(f)

    steps_list = pipeline_def.get("pipeline", pipeline_def).get("steps", pipeline_def.get("steps", []))
    step_map = {s["id"]: s["class"] for s in steps_list if "id" in s and "class" in s}

    if args.step_id not in step_map:
        print(f"Error: Unknown step '{args.step_id}'. Available: {list(step_map.keys())}", file=sys.stderr)
        return 1

    class_path = step_map[args.step_id]
    step_class = _resolve_step_class(class_path)

    # Load config: pipeline/config.yaml or --config, merge per-step + env
    config_path = args.config or (project_root / "pipeline" / "config.yaml")
    if not Path(config_path).exists():
        config_path = project_root / "config.yaml"
    full_config = _load_config(str(config_path))
    env = getattr(args, "env", None) or "dev"
    config = _merge_step_config(full_config, args.step_id, env)

    # Run context: from MLOPS_RUN_CONTEXT env or --run-context (cloud mode)
    run_context = None
    run_context_raw = os.environ.get("MLOPS_RUN_CONTEXT") or getattr(args, "run_context", None)
    if run_context_raw:
        run_context = _parse_run_context(run_context_raw)

    tracking = getattr(args, "tracking", False)
    tracking_backend = getattr(args, "tracking_backend", None) or full_config.get("tracking_backend", "noop")
    if tracking and tracking_backend == "noop":
        tracking_backend = "local"

    if run_context and run_context.base_path.startswith("gs://"):
        # Cloud mode: GCS base_path -> VertexRunner
        from urllib.parse import urlparse
        parsed = urlparse(run_context.base_path)
        bucket_name = parsed.netloc
        from mlops_framework.backends.execution.vertex_runner import VertexRunner
        runner = VertexRunner(
            bucket_name=bucket_name,
            experiment_name=run_context.experiment_name or "default",
            run_id=run_context.run_id,
        )
    else:
        # Local mode
        from mlops_framework.backends.execution.local_runner import LocalRunner
        runner = LocalRunner(
            artifacts_path=str(project_root / "artifacts"),
            runs_path=str(project_root / "runs"),
            tracking=tracking,
            tracking_backend=tracking_backend,
            run_context=run_context,
        )

    runner.run_step(step_class, config=config)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile pipeline YAML to Airflow DAG."""
    from mlops_framework.compiler.yaml_parser import parse_pipeline
    from mlops_framework.compiler.airflow_builder import build_airflow_dag

    pipeline_path = Path(args.pipeline_yaml)
    if not pipeline_path.exists():
        print(f"Error: Pipeline file not found: {pipeline_path}", file=sys.stderr)
        return 1

    with open(pipeline_path, "r") as f:
        raw = yaml.safe_load(f)

    pipeline_def = parse_pipeline(raw)
    dag_code = build_airflow_dag(pipeline_def)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dag_code)

    print(f"Generated Airflow DAG: {output_path}")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="mlops", description="MLOps framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # mlops run <step_id> --local [--config path] [--project-root path]
    run_parser = subparsers.add_parser("run", help="Run a pipeline step")
    run_parser.add_argument("step_id", help="Step ID from pipeline.yaml (e.g. preprocess, train)")
    run_parser.add_argument("--local", action="store_true", default=True, help="Run locally (default)")
    run_parser.add_argument("--env", default="dev", choices=["dev", "qa", "prod"], help="Environment (dev, qa, prod)")
    run_parser.add_argument("--config", help="Path to config YAML")
    run_parser.add_argument("--project-root", help="Project root (default: cwd)")
    run_parser.add_argument("--pipeline", default="pipeline/pipeline.yaml", help="Pipeline YAML path")
    run_parser.add_argument("--tracking", action="store_true", help="Enable experiment tracking (persist to ./runs)")
    run_parser.add_argument("--tracking-backend", choices=["noop", "local", "vertex"], help="Tracking backend: noop, local, or vertex")
    run_parser.add_argument("--run-context", help="RunContext JSON or path (cloud mode); also MLOPS_RUN_CONTEXT env")
    run_parser.set_defaults(func=cmd_run)

    # mlops compile <pipeline_yaml> --output <dag.py>
    compile_parser = subparsers.add_parser("compile", help="Compile pipeline to Airflow DAG")
    compile_parser.add_argument("pipeline_yaml", help="Path to pipeline YAML")
    compile_parser.add_argument("--output", "-o", required=True, help="Output DAG Python file")
    compile_parser.set_defaults(func=cmd_compile)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
