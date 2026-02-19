"""CLI entry point for mlplatform."""

import argparse
from pathlib import Path


def _load_train_data_from_csv(path: Path):
    """Load train data from CSV. Expects 'target' column for y, rest for X."""
    import pandas as pd

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"CSV must have 'target' column: {path}")
    X = df.drop(columns=["target"])
    y = df["target"]
    return {"X": X, "y": y}


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlplatform", description="MLOps framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run pipeline locally")
    run_parser.add_argument("--dag", required=True, help="DAG name or path (e.g. train_infer)")
    run_parser.add_argument("--env", default="dev", help="Environment (dev, qa, prod)")
    run_parser.add_argument("--steps-dir", help="Steps config directory (default: <dag_dir>/../steps)")
    run_parser.add_argument("--version", help="Model version (default: from config or 'dev')")
    run_parser.add_argument("--project-root", default="template_model", help="Model project root (e.g. template_model)")
    run_parser.add_argument("--train-data", help="Path to train CSV (must have 'target' column)")
    run_parser.add_argument("--inference-data", help="Path to inference CSV")

    build_parser = subparsers.add_parser("build-package", help="Build root.zip for Spark/Dataproc deployment")
    build_parser.add_argument("--model-package", default="template_model", help="Model package name")
    build_parser.add_argument("--output-dir", help="Output dir for root.zip (default: <project_root>/dist)")
    build_parser.add_argument("--project-root", default=".", help="Monorepo root (parent of model package)")

    run_spark_parser = subparsers.add_parser(
        "run-spark-main",
        help="Run main.py with root.zip (local simulation of Dataproc format)",
    )
    run_spark_parser.add_argument("--config", required=True, help="Path to run_config.json")
    run_spark_parser.add_argument("--packages", required=True, help="Path to root.zip")
    run_spark_parser.add_argument("--step-name", help="Step name override")
    run_spark_parser.add_argument("--input-path", help="Input CSV path (for train or inference)")

    args = parser.parse_args()

    if args.command == "run":
        from mlplatform.local import load_pipeline_config, run_pipeline_local

        project_root = Path(args.project_root).resolve()
        dag_arg = args.dag
        if dag_arg.endswith(".yaml") or dag_arg.endswith(".yml"):
            dag_path = project_root / dag_arg
        else:
            candidates = [
                project_root / "pipeline" / "dags" / f"{dag_arg}.yaml",
                Path("template_model") / "pipeline" / "dags" / f"{dag_arg}.yaml",
            ]
            dag_path = next((p for p in candidates if p.exists()), project_root / "pipeline" / "dags" / f"{dag_arg}.yaml")
        steps_dir = args.steps_dir
        if steps_dir is None:
            steps_dir = dag_path.parent.parent / "steps"
        else:
            steps_dir = project_root / steps_dir
        config = load_pipeline_config(
            dag_path=dag_path,
            steps_dir=steps_dir,
            env=args.env,
            version=args.version,
        )

        step_kwargs = {}
        if args.train_data:
            train_path = Path(args.train_data) if Path(args.train_data).is_absolute() else project_root / args.train_data
            step_kwargs["train"] = {"train_data": _load_train_data_from_csv(train_path)}
        if args.inference_data:
            import pandas as pd

            inf_path = Path(args.inference_data) if Path(args.inference_data).is_absolute() else project_root / args.inference_data
            step_kwargs["inference"] = {"inference_data": pd.read_csv(inf_path)}

        results = run_pipeline_local(config, project_root=project_root, base_path=project_root / "artifacts", **step_kwargs)
        for step_name, result in results.items():
            print(f"{step_name}: {result}")

    elif args.command == "build-package":
        from mlplatform.spark.packager import build_root_zip

        project_root = Path(args.project_root).resolve()
        output_dir = Path(args.output_dir) if args.output_dir else project_root / args.model_package / "dist"
        out = build_root_zip(
            project_root=project_root,
            model_package=args.model_package,
            output_dir=output_dir,
        )
        print(f"Built: {out}")

    elif args.command == "run-spark-main":
        from mlplatform.spark.main import run_spark_step

        result = run_spark_step(
            args.config,
            step_name=getattr(args, "step_name", None),
            packages=getattr(args, "packages", None),
            input_path=getattr(args, "input_path", None),
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
