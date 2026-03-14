"""Unified Spark entry point for both local and cloud execution.

Usage:
  Local (no zip):     python main.py --config <path.json>
  Local (with zip):   python main.py --config <path.json> --packages dist/root.zip
  Cloud spark-submit: spark-submit main.py --py-files dist/root.zip -- --config <path.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_local_paths(project_root: Path | None = None) -> None:
    """Add project_root and mlplatform to sys.path for local dev testing."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    for p in [str(project_root), str(project_root / "mlplatform")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _add_packages_to_path(packages: str) -> None:
    """Add zip packages to sys.path so model code can be imported."""
    for p in packages.split(","):
        p = p.strip()
        if not p:
            continue
        path = Path(p)
        if path.suffix == ".zip" and path.exists():
            sys.path.insert(0, str(path.resolve()))


def _load_config(config_path: str) -> dict[str, Any]:
    """Load run config from local file or GCS."""
    if config_path.startswith("gs://"):
        try:
            from google.cloud import storage as gcs
            parts = config_path[5:].split("/", 1)
            bucket_name, blob_path = parts[0], parts[1]
            client = gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            return json.loads(content)
        except ImportError:
            raise RuntimeError(
                "google-cloud-storage required for GCS config. pip install google-cloud-storage"
            )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="MLPlatform Spark entry point.")
    parser.add_argument("--config", required=True, help="Path to PipelineConfig JSON (local or gs://)")
    parser.add_argument("--input-path", help="Override input path from config")
    parser.add_argument("--output-path", help="Override output path from config")
    parser.add_argument("--packages", help="Comma-separated paths to zip packages")
    parser.add_argument("--project-root", help="Project root for local path bootstrap")
    args, _ = parser.parse_known_args()

    if args.packages:
        _add_packages_to_path(args.packages)
    else:
        project_root = Path(args.project_root).resolve() if args.project_root else None
        _bootstrap_local_paths(project_root)

    raw_config = _load_config(args.config)

    # V3: config is a PipelineConfig dict
    from mlplatform.config.models import PipelineConfig
    from mlplatform.runner.execute import execute

    # Apply CLI overrides
    if args.input_path:
        raw_config["input_path"] = args.input_path
    if args.output_path:
        raw_config["output_path"] = args.output_path

    config = PipelineConfig.from_dict(raw_config)

    try:
        result = execute(config)
        status = result.get("status", "unknown")
        if status.startswith("error"):
            print(f"Step failed: {status}", file=sys.stderr)
            return 1
        return 0
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
