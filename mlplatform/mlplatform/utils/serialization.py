"""Utilities for sanitizing Python data structures and converting to JSON-safe types.

These helpers have no mandatory dependencies beyond the Python standard library.
numpy and pandas types are handled via lazy imports when those packages are present.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime
from typing import Any


def sanitize(obj: Any) -> Any:
    """Recursively coerce *obj* to JSON-safe Python primitives.

    Handles:

    - ``numpy`` integer scalars → ``int``
    - ``numpy`` floating scalars → ``float`` (NaN/Inf → ``None``)
    - ``numpy.bool_`` → ``bool``
    - ``numpy.ndarray`` → ``list``
    - ``pandas.DataFrame`` → list of record dicts
    - ``pandas.Series`` → ``list``
    - ``float`` NaN / Inf → ``None``
    - ``datetime`` / ``date`` → ISO-format string
    - ``dict`` → recursively sanitized
    - ``list`` / ``tuple`` → recursively sanitized list
    - Everything else is returned unchanged.

    Examples::

        sanitize({"loss": float("nan"), "acc": 0.95})
        # → {"loss": None, "acc": 0.95}

        sanitize([np.int64(1), np.float32(2.5)])
        # → [1, 2.5]
    """
    # numpy types — lazy import so numpy is not a hard dependency
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [sanitize(item) for item in obj.tolist()]
    except ImportError:
        pass

    # pandas types — lazy import
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(obj, pd.DataFrame):
            return [sanitize(row) for row in obj.to_dict(orient="records")]
        if isinstance(obj, pd.Series):
            return [sanitize(item) for item in obj.tolist()]
    except ImportError:
        pass

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize(item) for item in obj]

    return obj


def to_serializable(obj: Any) -> Any:
    """Convert *obj* to a plain ``dict`` / ``list`` structure.

    Handles:

    - ``dataclasses.dataclass`` instances → ``dict`` (via ``dataclasses.asdict``)
    - Pydantic v2 models (`.model_dump()`) and v1 models (`.dict()`)
    - Arbitrary objects with ``__dict__`` → ``dict`` of public attributes
    - ``dict`` → recursively converted
    - ``list`` / ``tuple`` → recursively converted list
    - Primitive types (``str``, ``int``, ``float``, ``bool``, ``None``) →
      returned as-is

    This function only restructures objects into standard containers; it does
    *not* coerce numpy/pandas types. Compose with :func:`sanitize` when
    JSON-safe primitives are also required::

        result = sanitize(to_serializable(my_object))

    Examples::

        @dataclasses.dataclass
        class Metrics:
            accuracy: float
            loss: float

        to_serializable(Metrics(0.95, 0.12))
        # → {"accuracy": 0.95, "loss": 0.12}
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return obj.model_dump()

    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            result = obj.dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]

    if hasattr(obj, "__dict__"):
        return {
            k: to_serializable(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }

    return obj
