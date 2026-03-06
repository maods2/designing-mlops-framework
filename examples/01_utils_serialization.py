"""Example: mlplatform.utils — sanitize and to_serializable.

Install
-------
    pip install mlplatform[utils]
    # or, from this repo:
    pip install -e "mlplatform/[utils]"

Run
---
    python examples/01_utils_serialization.py
"""

import dataclasses
import sys
from datetime import date, datetime
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

from mlplatform.utils import sanitize, to_serializable  # noqa: E402


# ── 1. sanitize ─────────────────────────────────────────────────────────────

print("=" * 60)
print("1. sanitize — coerce non-JSON-safe values to safe Python types")
print("=" * 60)

# NaN and Inf become None
metrics = {
    "loss": float("nan"),
    "upper_bound": float("inf"),
    "accuracy": 0.95,
    "epoch": 42,
}
print("\nInput: ", metrics)
print("Output:", sanitize(metrics))
# → {'loss': None, 'upper_bound': None, 'accuracy': 0.95, 'epoch': 42}

# datetime / date → ISO string
event = {"trained_at": datetime(2024, 6, 1, 12, 0, 0), "date_only": date(2024, 6, 1)}
print("\nInput: ", event)
print("Output:", sanitize(event))
# → {'trained_at': '2024-06-01T12:00:00', 'date_only': '2024-06-01'}

# Nested structures are handled recursively
nested = {"outer": {"inner": [float("nan"), 1, float("inf")]}}
print("\nInput: ", nested)
print("Output:", sanitize(nested))
# → {'outer': {'inner': [None, 1, None]}}

# numpy types (if numpy is available)
try:
    import numpy as np

    np_metrics = {
        "count": np.int64(1000),
        "mean": np.float32(0.87),
        "nan_val": np.float64(float("nan")),
        "flags": np.array([True, False, True]),
    }
    print("\nInput (numpy): ", np_metrics)
    print("Output:        ", sanitize(np_metrics))
    # → {'count': 1000, 'mean': 0.87, 'nan_val': None, 'flags': [True, False, True]}
except ImportError:
    print("\n(numpy not installed — skipping numpy example)")


# ── 2. to_serializable ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. to_serializable — convert objects to plain dicts/lists")
print("=" * 60)


@dataclasses.dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


metrics_obj = ModelMetrics(accuracy=0.95, precision=0.93, recall=0.91, f1=0.92)
print("\nDataclass input: ", metrics_obj)
print("Output:          ", to_serializable(metrics_obj))
# → {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.91, 'f1': 0.92}


@dataclasses.dataclass
class TrainingRun:
    run_id: str
    metrics: ModelMetrics
    tags: list


run = TrainingRun(run_id="run-001", metrics=metrics_obj, tags=["v1", "prod"])
print("\nNested dataclass input: ", run)
print("Output:                 ", to_serializable(run))
# → {'run_id': 'run-001', 'metrics': {'accuracy': 0.95, ...}, 'tags': ['v1', 'prod']}


# ── 3. Composing sanitize + to_serializable ──────────────────────────────────

print("\n" + "=" * 60)
print("3. Compose both for fully JSON-ready output")
print("=" * 60)


@dataclasses.dataclass
class EvalResult:
    accuracy: float
    loss: float
    evaluated_at: datetime


result = EvalResult(accuracy=0.95, loss=float("nan"), evaluated_at=datetime(2024, 6, 1))
print("\nInput:  ", result)
print("Output:", sanitize(to_serializable(result)))
# → {'accuracy': 0.95, 'loss': None, 'evaluated_at': '2024-06-01T00:00:00'}
