"""Package model code as root.zip for Spark/Dataproc deployment."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence


def build_model_package(
    model_src: str | Path,
    output_path: str | Path,
    include_patterns: Sequence[str] = ("**/*.py", "**/*.yaml", "**/*.yml"),
    exclude_dirs: Sequence[str] = ("__pycache__", ".git", ".venv", "venv", ".pytest_cache", "dist"),
) -> Path:
    """Package model code into a zip file for Spark cluster deployment."""
    model_src = Path(model_src).resolve()
    output_path = Path(output_path)

    if not model_src.exists():
        raise FileNotFoundError(f"Model source not found: {model_src}")

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
    include_mlplatform: bool = True,
) -> Path:
    """Build root.zip containing model package and mlplatform framework.

    For cloud PySpark (Dataproc, VertexAI), workers need both the model code
    AND the mlplatform framework code in the zip.

    Standard layout in root.zip:
    - <model_package>/  (e.g. example_model/)
    - mlplatform/       (framework code, included for cloud workers)
    - config/           (optional)
    """
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / zip_name

    model_src = project_root / model_package
    if not model_src.exists():
        raise FileNotFoundError(f"Model package not found: {model_src}")

    exclude_dirs = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", "dist", ".egg-info"}
    include_tops = {model_package, "config"}
    if include_mlplatform:
        include_tops.add("mlplatform")

    seen: set[str] = set()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for top_name in include_tops:
            top_dir = project_root / top_name
            if not top_dir.exists():
                _try_installed_package(zf, top_name, seen, exclude_dirs)
                continue
            for fp in top_dir.rglob("*"):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(project_root)
                if any(ex in rel.parts for ex in exclude_dirs):
                    continue
                name = str(rel)
                if name not in seen:
                    seen.add(name)
                    zf.write(fp, name)

    return output_path


def _try_installed_package(
    zf: zipfile.ZipFile,
    package_name: str,
    seen: set[str],
    exclude_dirs: set[str],
) -> None:
    """If a package isn't in project_root, try to include it from the installed site-packages."""
    try:
        import importlib.util
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.origin is None:
            return
        pkg_root = Path(spec.origin).parent
        for fp in pkg_root.rglob("*"):
            if not fp.is_file():
                continue
            rel = Path(package_name) / fp.relative_to(pkg_root)
            if any(ex in rel.parts for ex in exclude_dirs):
                continue
            name = str(rel)
            if name not in seen:
                seen.add(name)
                zf.write(fp, name)
    except (ImportError, ValueError):
        pass
