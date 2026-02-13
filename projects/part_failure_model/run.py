"""Simple entry point for local development. Run steps from project root."""

import argparse
import os
import sys


def main():
    """Resolve step from pipeline/pipeline.yaml and run via framework."""
    from mlops_framework.cli.main import cmd_run

    parser = argparse.ArgumentParser()
    parser.add_argument("step_id", nargs="?", default="preprocess", help="Step ID (preprocess, train, inference)")
    parser.add_argument("--env", default="dev", choices=["dev", "qa", "prod"], help="Environment")
    parser.add_argument("--tracking", action="store_true", help="Enable experiment tracking (persist to ./runs)")
    parser.add_argument("--tracking-backend", choices=["noop", "local", "vertex"], help="Tracking backend")
    pargs = parser.parse_args()

    args = argparse.Namespace(
        step_id=pargs.step_id,
        local=True,
        env=pargs.env,
        config=None,
        project_root=os.getcwd(),
        pipeline="pipeline/pipeline.yaml",
        tracking=pargs.tracking,
        tracking_backend=getattr(pargs, "tracking_backend", None),
        run_context=None,
    )
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
