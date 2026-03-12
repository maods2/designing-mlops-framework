"""Example: sanitize and to_serializable.

Run: python examples/01_utils_serialization.py

Note: When using artifact.save("metrics.json", metrics), the artifact registry
sanitizes dicts automatically. Use sanitize/to_serializable when building data
for other purposes (e.g. custom JSON, HTML reports).
"""

import dataclasses
from datetime import datetime

import _bootstrap  # noqa: F401

from mlplatform.utils import sanitize, to_serializable

# sanitize: NaN/Inf → None, datetime → ISO string
print(sanitize({"loss": float("nan"), "accuracy": 0.95}))
print(sanitize({"trained_at": datetime(2024, 6, 1, 12, 0, 0)}))

# to_serializable: dataclass → dict
@dataclasses.dataclass
class Metrics:
    accuracy: float
    loss: float

print(to_serializable(Metrics(0.95, 0.12)))
print(sanitize(to_serializable(Metrics(0.95, float("nan")))))
