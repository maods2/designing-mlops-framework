"""Example: mlplatform.artifacts — create_artifacts standalone and config-driven.

Demonstrates create_artifacts() in two modes:
  1. Standalone — explicit params (local or GCS)
  2. Config-driven — from workflow + model config

Install
-------
    pip install mlplatform[core]
    # or, from this repo:
    pip install -e "mlplatform/[core]"

Run
---
    python examples/05_artifacts_standalone.py

Output is written to examples/output/artifacts_demo/ — safe to delete.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — only needed when running directly from the repo without
# a pip install.  Safe to remove if the package is installed.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
_mlplatform_src = _repo_root / "mlplatform"
for _p in [str(_repo_root), str(_mlplatform_src)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from mlplatform.artifacts import create_artifacts  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "artifacts_demo"


# ── 1. Standalone mode (local) ───────────────────────────────────────────────

print("=" * 60)
print("1. Standalone mode — local backend")
print("=" * 60)

artifacts = create_artifacts(
    backend="local",
    base_path=str(OUTPUT_DIR),
    feature_name="demo",
    model_name="sample_model",
    version="v1",
)

# Show resolve_path convention: {feature}/{model}/{version}/{name}
path = artifacts.resolve_path("model.pkl")
print(f"\nArtifact path convention: {path}")

# Save and load
model_data = {"weights": [1.0, 2.0], "bias": 0.5}
artifacts.save("model.pkl", model_data)
loaded = artifacts.load("model.pkl")
print(f"Saved: {model_data}")
print(f"Loaded: {loaded}")
print(f"Match: {loaded == model_data}")


# ── 2. Standalone mode (GCS) — commented ────────────────────────────────────

print("\n" + "=" * 60)
print("2. Standalone mode — GCS backend (commented)")
print("=" * 60)

print("""
# For GCS, use:
#
#   artifacts = create_artifacts(
#       backend="gcs",
#       bucket="my-bucket",
#       prefix="models",
#       project="my-gcp-project",
#       feature_name="feature",
#       model_name="model",
#       version="v1",
#   )
#   artifacts.save("model.pkl", model)
#
# Requires: pip install mlplatform[storage]
""")


# ── 3. Config-driven mode ───────────────────────────────────────────────────

print("=" * 60)
print("3. Config-driven mode — from workflow + model config")
print("=" * 60)

from mlplatform.config.loader import load_workflow_config  # noqa: E402

train_dag = _repo_root / "example_model_v2" / "pipeline" / "train.yaml"
if train_dag.exists():
    workflow = load_workflow_config(train_dag)
    model_cfg = workflow.models[0]

    config_artifacts = create_artifacts(
        workflow=workflow,
        model_cfg=model_cfg,
        version="dev",
        profile="local",
        base_path_override=str(OUTPUT_DIR / "config_driven"),
    )

    path = config_artifacts.resolve_path("config_demo.pkl")
    print(f"\nArtifact path: {path}")

    config_artifacts.save("config_demo.pkl", {"source": "config_driven"})
    reloaded = config_artifacts.load("config_demo.pkl")
    print(f"Loaded: {reloaded}")
else:
    print("\n(Skipping — example_model_v2 not found)")

print("\nDone. Output written to:", OUTPUT_DIR)
