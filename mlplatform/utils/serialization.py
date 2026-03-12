"""Utilities for sanitizing Python data structures and converting to JSON-safe types.

Used internally by :class:`~mlplatform.artifacts.registry.ArtifactRegistry` when
saving dicts to ``.json`` files — you do not need to call :func:`sanitize` before
:meth:`~mlplatform.artifacts.registry.ArtifactRegistry.save` for JSON.

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

    Handles dataclasses, Pydantic models, objects with ``__dict__``.
    Compose with :func:`sanitize` when JSON-safe primitives are also required.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return obj.model_dump()

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
