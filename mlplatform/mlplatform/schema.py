"""Schema module for batch prediction input validation.

Provides a simple way for data scientists to define clear input schemas for
batch prediction. The framework validates data automatically before calling
predict() — no manual validate() call needed.

Simplest usage (zero schema code): define FEATURE_COLUMNS in constants.py.
The framework auto-derives validation from it.

Explicit schema (when you need dtype checks or optional columns)::

    from mlplatform.schema import PredictionInputSchema

    INPUT_SCHEMA = PredictionInputSchema(columns=["f0", "f1", "f2", "f3", "f4"])
"""

from __future__ import annotations

from typing import Any, Sequence


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not match the expected schema."""


def from_feature_columns(column_names: Sequence[str]) -> "PredictionInputSchema":
    """Build a minimal schema from a list of required column names.

    Use this when your predictor uses FEATURE_COLUMNS from constants.py.
    The framework can auto-derive validation from this.

    Example::

        from mlplatform.schema import from_feature_columns
        INPUT_SCHEMA = from_feature_columns(["f0", "f1", "f2", "f3", "f4"])
    """
    return PredictionInputSchema(columns=list(column_names))


def get_schema_from_predictor(predictor: Any) -> PredictionInputSchema | None:
    """Get validation schema from a predictor instance.

    Looks for INPUT_SCHEMA in the predictor's module first. If not found,
    tries to import the predictor's constants module (e.g. example_model.constants)
    and build a schema from FEATURE_COLUMNS.

    Returns:
        A PredictionInputSchema to validate input data, or None if no schema
        can be derived.
    """
    import importlib

    predictor_cls = type(predictor)
    module_name = predictor_cls.__module__

    # 1. Check for explicit INPUT_SCHEMA in predictor module
    try:
        mod = importlib.import_module(module_name)
        schema = getattr(mod, "INPUT_SCHEMA", None)
        if isinstance(schema, PredictionInputSchema):
            return schema
    except ImportError:
        pass

    # 2. Try constants module (e.g. example_model.constants from example_model.predict)
    parts = module_name.split(".")
    if len(parts) >= 2:
        constants_name = f"{parts[0]}.constants"
        try:
            constants_mod = importlib.import_module(constants_name)
            feature_cols = getattr(constants_mod, "FEATURE_COLUMNS", None)
            if isinstance(feature_cols, (list, tuple)) and feature_cols:
                return from_feature_columns(feature_cols)
        except ImportError:
            pass

    return None


class PredictionInputSchema:
    """Define and validate the input schema for batch prediction.

    Each column entry is a tuple of ``(name, dtype, required)`` where:
    - ``name``: column name (str)
    - ``dtype``: expected dtype string (e.g. "float64", "int64", "object") — ``None`` skips dtype check
    - ``required``: if ``True``, the column must be present

    Simpler form — just a list of column names (all required, no dtype check)::

        schema = PredictionInputSchema(columns=["f0", "f1", "f2"])
    """

    def __init__(
        self,
        columns: Sequence[Any],
        strict: bool = False,
    ) -> None:
        """
        Args:
            columns: Either a list of column name strings, or a list of
                     ``(name, dtype, required)`` tuples.
            strict: If ``True``, raise an error if the DataFrame contains
                    columns not declared in the schema.
        """
        self._columns: list[tuple[str, str | None, bool]] = []
        for col in columns:
            if isinstance(col, str):
                self._columns.append((col, None, True))
            elif isinstance(col, (list, tuple)) and len(col) == 3:
                self._columns.append((str(col[0]), col[1], bool(col[2])))
            elif isinstance(col, (list, tuple)) and len(col) == 2:
                self._columns.append((str(col[0]), col[1], True))
            else:
                raise ValueError(f"Invalid column spec: {col!r}. Expected str or (name, dtype, required) tuple.")
        self.strict = strict

    @property
    def column_names(self) -> list[str]:
        """Return all declared column names."""
        return [name for name, _, _ in self._columns]

    def validate(self, data: Any) -> None:
        """Validate *data* (a pandas DataFrame) against the schema.

        Args:
            data: A pandas DataFrame to validate.

        Raises:
            SchemaValidationError: If required columns are missing, dtype mismatches
                occur, or (when strict=True) undeclared columns are present.
            ImportError: If pandas is not installed.
        """
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise SchemaValidationError(
                f"Expected a pandas DataFrame, got {type(data).__name__}."
            )

        errors: list[str] = []
        df_cols = set(data.columns)

        for col_name, col_dtype, required in self._columns:
            if col_name not in df_cols:
                if required:
                    errors.append(f"Missing required column: '{col_name}'")
            elif col_dtype is not None:
                actual_dtype = str(data[col_name].dtype)
                if actual_dtype != col_dtype:
                    errors.append(
                        f"Column '{col_name}' has dtype '{actual_dtype}', expected '{col_dtype}'"
                    )

        if self.strict:
            declared = set(self.column_names)
            extra = df_cols - declared
            if extra:
                errors.append(f"Undeclared columns (strict mode): {sorted(extra)}")

        if errors:
            raise SchemaValidationError(
                "Schema validation failed:\n  " + "\n  ".join(errors)
            )

    def __repr__(self) -> str:
        cols = ", ".join(
            f"{name}({dtype or 'any'}, {'required' if req else 'optional'})"
            for name, dtype, req in self._columns
        )
        return f"PredictionInputSchema([{cols}], strict={self.strict})"
