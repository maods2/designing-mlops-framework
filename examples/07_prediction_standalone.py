"""Example: Prediction with dev_predict and BasePredictor.

Demonstrates:
  1. One-liner prediction via dev_predict (loads model, runs on configured input)
  2. dev_predict with manual DataFrame (skip file I/O)

Install
-------
    pip install mlplatform[core]
    pip install -r example_model/requirements.txt   # sklearn for example_model

Prerequisites
-------------
    Run 06_train_standalone.py or example_model/train.py first to create a model.

Run
---
    python examples/07_prediction_standalone.py

Output is written to examples/output/prediction_demo/ — safe to delete.
"""

import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "prediction_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1. dev_predict with manual DataFrame (no file I/O)")
print("=" * 60)

pred_dag = _repo_root / "example_model" / "pipeline" / "predict.yaml"
if pred_dag.exists():
    from mlplatform.runner import dev_predict

    # Create sample data (same columns as example_model expects)
    sample_df = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]],
        columns=["f0", "f1", "f2", "f3", "f4"],
    )

    result = dev_predict(
        pred_dag,
        data=sample_df,
        profile="local",
        version="demo_v1",
        base_path=str(OUTPUT_DIR.parent / "train_demo"),
    )
    if result is not None:
        print(f"\nPredictions:\n{result}")
        print(f"Columns: {list(result.columns)}")
    else:
        print("\n(No result — model may not exist. Run 06_train_standalone.py first.)")
else:
    print("\n(Skipping — example_model not found)")

print("\n" + "=" * 60)
print("2. dev_predict from file (uses input_path from config)")
print("=" * 60)

if pred_dag.exists():
    from mlplatform.runner import dev_predict

    # When data=None, dev_predict loads from model_cfg.input_path
    try:
        result = dev_predict(
            pred_dag,
            data=None,
            profile="local",
            version="demo_v1",
            base_path=str(OUTPUT_DIR.parent / "train_demo"),
        )
        if result is not None:
            print(f"\nLoaded from config input_path, got {len(result)} rows")
    except Exception as e:
        print(f"\n(File-based prediction: {e})")
        print("  Ensure example_model/data/sample_inference.csv exists and model is trained.")

print("\nDone. Output written to:", OUTPUT_DIR)
