"""CLI entry point for mlplatform."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlplatform", description="MLOps framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run: execute a workflow from a DAG YAML ---
    run_parser = subparsers.add_parser("run", help="Run workflow locally")
    run_parser.add_argument("--dag", required=True, help="Path to DAG YAML config")
    run_parser.add_argument(
        "--profile", default="local",
        help="Profile name (local, local-spark, cloud-batch, cloud-online, cloud-batch-emulated)",
    )
    run_parser.add_argument("--version", help="Model version (default: auto-generated)")
    run_parser.add_argument("--project-root", default=".", help="Project root directory")
    run_parser.add_argument("--base-path", help="Artifact storage base path (default: ./artifacts)")

    # --- build-package: create root.zip for cloud deployment ---
    build_parser = subparsers.add_parser("build-package", help="Build root.zip for Spark/Dataproc deployment")
    build_parser.add_argument("--model-package", default="example_model", help="Model package name")
    build_parser.add_argument("--output-dir", help="Output dir for root.zip")
    build_parser.add_argument("--project-root", default=".", help="Project root directory")
    build_parser.add_argument(
        "--include-mlplatform", action="store_true", default=True,
        help="Include mlplatform framework in root.zip (required for cloud)",
    )

    # --- profiles: list available profiles ---
    subparsers.add_parser("profiles", help="List available execution profiles")

    args = parser.parse_args()

    if args.command == "run":
        from mlplatform.local import run_workflow

        project_root = Path(args.project_root).resolve()
        dag_path = Path(args.dag)
        if not dag_path.is_absolute():
            dag_path = project_root / dag_path

        results = run_workflow(
            dag_path=dag_path,
            profile_name=args.profile,
            version=args.version,
            project_root=project_root,
            base_path=args.base_path,
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
            include_mlplatform=args.include_mlplatform,
        )
        print(f"Built: {out}")

    elif args.command == "profiles":
        from mlplatform.profiles.registry import list_profiles

        profiles = list_profiles()
        print("Available profiles:")
        for p in profiles:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
