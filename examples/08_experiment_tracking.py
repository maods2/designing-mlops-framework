"""Example: Experiment tracking with LocalJsonTracker and ExecutionContext.

Demonstrates:
  1. LocalJsonTracker directly — log params, metrics, persist to JSON
  2. Tracker via ExecutionContext (as used by BaseTrainer)
  3. Profile-driven tracker (local profile uses LocalJsonTracker)

Install
-------
    pip install mlplatform[core]

Run
---
    python examples/08_experiment_tracking.py

Output is written to examples/output/tracking_demo/ — safe to delete.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "tracking_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1. LocalJsonTracker directly")
print("=" * 60)

from mlplatform.tracking.local import LocalJsonTracker

tracker = LocalJsonTracker(base_path=str(OUTPUT_DIR), run_id="run_001")
tracker.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
tracker.log_params({"random_state": 42})
tracker.log_metrics({"accuracy": 0.92, "f1": 0.89})
tracker.log_metrics({"precision": 0.91, "recall": 0.88})

metrics_path = OUTPUT_DIR / "metrics.json"
tracker.save(str(metrics_path))
print(f"\nSaved to {metrics_path}")
print(metrics_path.read_text())

print("\n" + "=" * 60)
print("2. Tracker via ExecutionContext (as BaseTrainer sees it)")
print("=" * 60)

from mlplatform.profiles.registry import get_profile
from mlplatform.core.context import ExecutionContext

prof = get_profile("local")
ctx = ExecutionContext.from_profile(
    profile=prof,
    feature_name="demo",
    model_name="tracking_example",
    version="v1",
    base_path=str(OUTPUT_DIR),
)

# Same interface as BaseTrainer.tracker
ctx.experiment_tracker.log_params({"experiment": "demo", "phase": "test"})
ctx.experiment_tracker.log_metrics({"loss": 0.05, "val_loss": 0.07})

run_path = OUTPUT_DIR / "demo" / "tracking_example" / "v1"
run_path.mkdir(parents=True, exist_ok=True)
ctx.experiment_tracker.save(str(run_path / "run.json"))

print(f"\nContext tracker type: {type(ctx.experiment_tracker).__name__}")
print(f"Saved to {run_path / 'run.json'}")

print("\n" + "=" * 60)
print("3. NoneTracker (no-op, for environments without tracking)")
print("=" * 60)

from mlplatform.tracking.none import NoneTracker

noop = NoneTracker()
noop.log_params({"foo": "bar"})
noop.log_metrics({"acc": 1.0})
# No output — never fails, never persists
print("\nNoneTracker: log_params/log_metrics are no-ops (no output)")

print("\nDone. Output written to:", OUTPUT_DIR)
