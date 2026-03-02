"""CLI entry point for mlplatform."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlplatform", description="MLOps framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run workflow locally")
    run_parser.add_argument("--dag", required=True, help="Path to DAG YAML config")
    run_parser.add_argument(
        "--profile", default="local",
        help="Profile name (local, local-spark, cloud-train, cloud-online, cloud-batch, cloud-batch-emulated)",
    )
    run_parser.add_argument("--version", help="Model version (default: auto-generated)")
    run_parser.add_argument("--base-path", help="Artifact storage base path (default: ./artifacts)")
    run_parser.add_argument("--commit-hash", help="Git commit hash to track the code version used for this run")
    run_parser.add_argument(
        "--config",
        help="Comma-separated config profile names to load and merge (e.g. global,local). "
             "Overrides the 'config:' key declared in the DAG YAML.",
    )

    build_parser = subparsers.add_parser("build-package", help="Build root.zip for Spark/Dataproc deployment")
    build_parser.add_argument("--model-package", default="example_model", help="Model package name")
    build_parser.add_argument("--output-dir", help="Output dir for root.zip")
    build_parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    if args.command == "run":
        from mlplatform.runner import run_workflow

        dag_path = Path(args.dag)
        config_names = None
        if args.config:
            config_names = [c.strip() for c in args.config.split(",") if c.strip()]
        results = run_workflow(
            dag_path=dag_path,
            profile=args.profile,
            version=args.version,
            base_path=args.base_path,
            commit_hash=args.commit_hash,
            config_names=config_names,
        )
        for model_name, result in results.items():
            print(f"{model_name}: {result}")

    elif args.command == "build-package":
        from mlplatform.spark.packager import build_root_zip

        project_root = Path(args.project_root).resolve()
        output_dir = Path(args.output_dir) if args.output_dir else project_root / "dist"
        out = build_root_zip(
            project_root=project_root,
            model_package=args.model_package,
            output_dir=output_dir,
        )
        print(f"Built: {out}")


if __name__ == "__main__":
    main()
