"""CLI entry point for mlplatform — args -> PipelineConfig -> execute()."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlplatform", description="MLOps framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run a single model pipeline")
    run_parser.add_argument("--model-name", required=True, help="Model identifier")
    run_parser.add_argument("--feature", required=True, help="Feature domain")
    run_parser.add_argument("--pipeline-type", required=True, choices=["training", "prediction"],
                            help="Pipeline type")
    run_parser.add_argument("--module", default="", help="Module path (e.g. model_code.train:MyTrainer)")
    run_parser.add_argument(
        "--profile", default="local",
        help="Profile name (local, cloud-train, cloud-batch, etc.)",
    )
    run_parser.add_argument("--version", default="dev", help="Model version")
    run_parser.add_argument("--base-path", default="./artifacts", help="Artifact storage base path")
    run_parser.add_argument("--backend", default="local", choices=["local", "gcs"],
                            help="Storage backend")
    run_parser.add_argument("--base-bucket", help="GCS bucket name")
    run_parser.add_argument("--project-id", help="GCP project ID")
    run_parser.add_argument("--commit-hash", help="Git commit hash for tracking")
    run_parser.add_argument(
        "--config", default="global,dev",
        help="Comma-separated config profile names (e.g. global,dev)",
    )
    run_parser.add_argument("--config-dir", default="./config", help="Config profiles directory")
    run_parser.add_argument("--log-level", default="INFO", help="Log level")

    # --- build-package command ---
    build_parser = subparsers.add_parser("build-package", help="Build root.zip for Spark/Dataproc")
    build_parser.add_argument("--model-package", default="model_code", help="Model package name")
    build_parser.add_argument("--output-dir", help="Output dir for root.zip")
    build_parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    if args.command == "run":
        from mlplatform.config.builder import PipelineConfigBuilder
        from mlplatform.runner.execute import execute

        config_list = [c.strip() for c in args.config.split(",") if c.strip()]

        builder = (
            PipelineConfigBuilder()
            .identity(model_name=args.model_name, feature=args.feature, version=args.version)
            .infra(
                backend=args.backend,
                base_path=args.base_path,
                base_bucket=args.base_bucket,
                project_id=args.project_id,
            )
            .pipeline(
                pipeline_type=args.pipeline_type,
                profile=args.profile,
                module=args.module,
            )
            .configs(config_list, config_dir=args.config_dir)
            .metadata(commit_hash=args.commit_hash, log_level=args.log_level)
        )

        config = builder.build()
        result = execute(config)
        print(f"{result.get('model_name', 'unknown')}: {result.get('status', 'unknown')}")

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
