"""Schema module for batch prediction input validation.

Provides a simple way for data scientists to define clear input schemas for
batch prediction. The framework validates data before calling predict().

Example usage::

    from mlplatform.schema import PredictionInputSchema

    INPUT_SCHEMA = PredictionInputSchema(
        columns=[
            ("f0", "float64", True),
            ("f1", "float64", True),
            ("f2", "float64", True),
            ("f3", "float64", True),
            ("f4", "float64", True),
        ]
    )

    # In predictor.predict():
    INPUT_SCHEMA.validate(data)
"""

from __future__ import annotations

from typing import Any, Sequence


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not match the expected schema."""


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
