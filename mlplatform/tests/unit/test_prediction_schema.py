"""Unit tests for mlplatform.schema (PredictionInputSchema)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlplatform.schema import (
    PredictionInputSchema,
    SchemaValidationError,
    from_feature_columns,
    get_schema_from_predictor,
)


class TestPredictionInputSchemaInit:
    def test_string_columns(self):
        schema = PredictionInputSchema(columns=["f0", "f1", "f2"])
        assert schema.column_names == ["f0", "f1", "f2"]

    def test_tuple_columns(self):
        schema = PredictionInputSchema(
            columns=[("f0", "float64", True), ("label", "int64", False)]
        )
        assert schema.column_names == ["f0", "label"]

    def test_invalid_column_spec(self):
        with pytest.raises(ValueError, match="Invalid column spec"):
            PredictionInputSchema(columns=[42])

    def test_repr(self):
        schema = PredictionInputSchema(columns=["f0"])
        assert "PredictionInputSchema" in repr(schema)


class TestPredictionInputSchemaValidate:
    def test_valid_df_string_columns(self):
        schema = PredictionInputSchema(columns=["f0", "f1"])
        df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
        schema.validate(df)  # should not raise

    def test_valid_df_tuple_columns(self):
        schema = PredictionInputSchema(
            columns=[("f0", "float64", True), ("f1", "float64", True)]
        )
        df = pd.DataFrame({"f0": [1.0], "f1": [2.0]}, dtype="float64")
        schema.validate(df)  # should not raise

    def test_missing_required_column(self):
        schema = PredictionInputSchema(columns=["f0", "f1", "f2"])
        df = pd.DataFrame({"f0": [1.0], "f1": [2.0]})
        with pytest.raises(SchemaValidationError, match="Missing required column: 'f2'"):
            schema.validate(df)

    def test_optional_column_absent_ok(self):
        schema = PredictionInputSchema(columns=[("f0", None, True), ("opt", None, False)])
        df = pd.DataFrame({"f0": [1.0]})
        schema.validate(df)  # optional column absent — should not raise

    def test_dtype_mismatch(self):
        schema = PredictionInputSchema(columns=[("f0", "float64", True)])
        df = pd.DataFrame({"f0": ["a", "b"]})
        with pytest.raises(SchemaValidationError, match="dtype"):
            schema.validate(df)

    def test_strict_mode_extra_columns(self):
        schema = PredictionInputSchema(columns=["f0"], strict=True)
        df = pd.DataFrame({"f0": [1.0], "extra": [2.0]})
        with pytest.raises(SchemaValidationError, match="Undeclared columns"):
            schema.validate(df)

    def test_non_dataframe_raises(self):
        schema = PredictionInputSchema(columns=["f0"])
        with pytest.raises(SchemaValidationError, match="Expected a pandas DataFrame"):
            schema.validate({"f0": [1.0]})

    def test_multiple_errors_reported(self):
        schema = PredictionInputSchema(columns=["f0", "f1", "f2"])
        df = pd.DataFrame({"other": [1.0]})
        with pytest.raises(SchemaValidationError) as exc_info:
            schema.validate(df)
        msg = str(exc_info.value)
        assert "f0" in msg
        assert "f1" in msg
        assert "f2" in msg


class TestFromFeatureColumns:
    def test_builds_schema_from_list(self):
        schema = from_feature_columns(["f0", "f1", "f2"])
        assert schema.column_names == ["f0", "f1", "f2"]
        df = pd.DataFrame({"f0": [1.0], "f1": [2.0], "f2": [3.0]})
        schema.validate(df)  # should not raise


class TestGetSchemaFromPredictor:
    def test_gets_schema_from_feature_columns(self):
        from example_model.predict import MyPredictor

        predictor = MyPredictor()
        schema = get_schema_from_predictor(predictor)
        assert schema is not None
        assert schema.column_names == ["f0", "f1", "f2", "f3", "f4"]

    def test_returns_none_when_no_schema(self):
        class MinimalPredictor:
            __module__ = "nonexistent.module"

        schema = get_schema_from_predictor(MinimalPredictor())
        assert schema is None
