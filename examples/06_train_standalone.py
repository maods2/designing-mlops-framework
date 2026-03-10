"""Example: Training with dev_train and BaseTrainer.

Demonstrates:
  1. One-liner training via dev_train (loads DAG, builds context, runs trainer)
  2. Manual flow: dev_context + custom trainer (for custom setup)

Install
-------
    pip install mlplatform[core]
    pip install -r example_model/requirements.txt   # sklearn for example_model

Run
---
    python examples/06_train_standalone.py

Output is written to examples/output/train_demo/ — safe to delete.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "train_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1. One-liner training via dev_train")
print("=" * 60)

# Use example_model if available
train_dag = _repo_root / "example_model" / "pipeline" / "train.yaml"
if train_dag.exists():
    from mlplatform.runner import dev_train

    trainer = dev_train(
        train_dag,
        profile="local",
        version="demo_v1",
        base_path=str(OUTPUT_DIR),
    )
    print(f"\nTrained: {trainer.context.model_name}")
    print(f"Artifacts at: {OUTPUT_DIR / 'example' / 'example_model' / 'demo_v1'}")
else:
    print("\n(Skipping — example_model not found)")

print("\n" + "=" * 60)
print("2. Manual flow: dev_context + minimal trainer")
print("=" * 60)

from mlplatform.core.trainer import BaseTrainer
from mlplatform.runner import dev_context

# Use a minimal DAG or create context manually
# For standalone demo, we use a minimal trainer with explicit context
dag = _repo_root / "example_model_v2" / "pipeline" / "train.yaml"
if not dag.exists():
    dag = train_dag  # fallback to example_model

if dag.exists():
    ctx = dev_context(
        dag,
        profile="local",
        version="manual_demo",
        base_path=str(OUTPUT_DIR),
    )

    class MinimalTrainer(BaseTrainer):
        """Minimal trainer — saves a dummy artifact to show the flow."""

        def train(self) -> None:
            self.tracker.log_params({"model_type": "minimal", "demo": True})
            self.tracker.log_metrics({"accuracy": 0.99})
            self.artifacts.save("demo_model.pkl", {"weights": [1.0, 2.0], "bias": 0.5})
            self.log.info("Minimal training complete")

    trainer = MinimalTrainer()
    trainer.context = ctx
    trainer.setup()
    trainer.train()
    trainer.teardown()
    print(f"\nSaved demo_model.pkl to {ctx.model_name}/manual_demo/")
else:
    print("\n(Skipping — no pipeline DAG found)")

print("\nDone. Output written to:", OUTPUT_DIR)
