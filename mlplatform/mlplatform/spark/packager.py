"""Package model code as root.zip for Spark/Dataproc deployment."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence


def build_model_package(
    model_src: str | Path,
    output_path: str | Path,
    include_patterns: Sequence[str] = ("**/*.py", "**/*.yaml", "**/*.yml"),
    exclude_dirs: Sequence[str] = ("__pycache__", ".git", ".venv", "venv", ".pytest_cache"),
) -> Path:
    """
    Package model code into a zip file for Spark cluster deployment.

    Args:
        model_src: Path to model package (e.g. template_model/ or project root)
        output_path: Output zip path (e.g. root.zip, gs://bucket/root.zip for GCS)
        include_patterns: Glob patterns for files to include
        exclude_dirs: Directory names to exclude

    Returns:
        Path to the created zip file
    """
    model_src = Path(model_src).resolve()
    output_path = Path(output_path)

    if not model_src.exists():
        raise FileNotFoundError(f"Model source not found: {model_src}")

    # For local output, ensure parent exists
    if not str(output_path).startswith("gs://"):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in include_patterns:
            for fp in model_src.glob(pattern):
                if fp.is_file():
                    rel = fp.relative_to(model_src)
                    if any(ex in rel.parts for ex in exclude_dirs):
                        continue
                    zf.write(fp, str(rel))

    return output_path


def build_root_zip(
    project_root: str | Path,
    model_package: str,
    output_dir: str | Path = "./dist",
    zip_name: str = "root.zip",
) -> Path:
    """
    Build root.zip containing model package and pipeline configs.

    Standard layout for Dataproc:
    - root.zip
      - template_model/  (or model_package name)
      - config/          (optional env configs)
      - pipeline/        (optional DAG/step YAMLs)

    Args:
        project_root: Project root directory
        model_package: Name of model package (e.g. template_model)
        output_dir: Where to write root.zip
        zip_name: Output zip filename

    Returns:
        Path to root.zip
    """
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / zip_name

    model_src = project_root / model_package
    if not model_src.exists():
        raise FileNotFoundError(f"Model package not found: {model_src}")

    seen = set()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in project_root.rglob("*"):
            if not fp.is_file() or "__pycache__" in str(fp):
                continue
            rel = fp.relative_to(project_root)
            # Only include model package, config, and pipeline
            top = rel.parts[0] if rel.parts else ""
            if top not in (model_package, "config"):
                continue
            name = str(rel)
            if name in seen:
                continue
            seen.add(name)
            zf.write(fp, name)

    return output_path
