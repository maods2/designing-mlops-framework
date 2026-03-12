"""Path bootstrap for running examples from repo root. Remove if package is installed."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for p in (str(_root), str(_root / "mlplatform")):
    if p not in sys.path:
        sys.path.insert(0, p)
