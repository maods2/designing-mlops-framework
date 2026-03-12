"""Unit tests for mlplatform.utils."""

from __future__ import annotations

import dataclasses
import io
import math
from datetime import date, datetime

import pytest

from mlplatform.storage.local import LocalFileSystem
from mlplatform.utils import HTMLReport, sanitize, save_html, save_plot, to_serializable


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_float_nan_becomes_none(self):
        assert sanitize(float("nan")) is None

    def test_float_positive_inf_becomes_none(self):
        assert sanitize(float("inf")) is None

    def test_float_negative_inf_becomes_none(self):
        assert sanitize(float("-inf")) is None

    def test_regular_float_unchanged(self):
        assert sanitize(3.14) == pytest.approx(3.14)

    def test_nested_dict_with_nan(self):
        result = sanitize({"loss": float("nan"), "acc": 0.95})
        assert result == {"loss": None, "acc": 0.95}

    def test_list_with_mixed_values(self):
        result = sanitize([1, float("nan"), "x", None])
        assert result == [1, None, "x", None]

    def test_tuple_coerced_to_list(self):
        result = sanitize((1, 2, 3))
        assert result == [1, 2, 3]

    def test_datetime_to_iso(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert sanitize(dt) == "2024-01-15T10:30:00"

    def test_date_to_iso(self):
        d = date(2024, 6, 1)
        assert sanitize(d) == "2024-06-01"

    def test_primitives_passthrough(self):
        assert sanitize(42) == 42
        assert sanitize("hello") == "hello"
        assert sanitize(True) is True
        assert sanitize(None) is None

    def test_nested_structure(self):
        obj = {"metrics": {"val_loss": float("nan")}, "epoch": 5}
        result = sanitize(obj)
        assert result == {"metrics": {"val_loss": None}, "epoch": 5}

    def test_numpy_integer(self):
        np = pytest.importorskip("numpy")
        assert sanitize(np.int64(7)) == 7
        assert isinstance(sanitize(np.int32(3)), int)

    def test_numpy_floating(self):
        np = pytest.importorskip("numpy")
        result = sanitize(np.float32(1.5))
        assert isinstance(result, float)
        assert result == pytest.approx(1.5, abs=1e-4)

    def test_numpy_float_nan_becomes_none(self):
        np = pytest.importorskip("numpy")
        assert sanitize(np.float64("nan")) is None

    def test_numpy_bool(self):
        np = pytest.importorskip("numpy")
        assert sanitize(np.bool_(True)) is True
        assert isinstance(sanitize(np.bool_(False)), bool)

    def test_numpy_array(self):
        np = pytest.importorskip("numpy")
        result = sanitize(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([1, 2, 3])
        assert sanitize(s) == [1, 2, 3]

    def test_pandas_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = sanitize(df)
        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


# ---------------------------------------------------------------------------
# to_serializable
# ---------------------------------------------------------------------------


class TestToSerializable:
    def test_dataclass_flat(self):
        @dataclasses.dataclass
        class Point:
            x: float
            y: float

        result = to_serializable(Point(1.0, 2.0))
        assert result == {"x": 1.0, "y": 2.0}

    def test_nested_dataclasses(self):
        @dataclasses.dataclass
        class Inner:
            val: int

        @dataclasses.dataclass
        class Outer:
            inner: Inner

        result = to_serializable(Outer(Inner(99)))
        assert result == {"inner": {"val": 99}}

    def test_dict_recursion(self):
        @dataclasses.dataclass
        class Item:
            name: str

        d = {"item": Item("widget"), "count": 3}
        result = to_serializable(d)
        assert result == {"item": {"name": "widget"}, "count": 3}

    def test_list_of_dataclasses(self):
        @dataclasses.dataclass
        class Pt:
            x: int

        result = to_serializable([Pt(1), Pt(2)])
        assert result == [{"x": 1}, {"x": 2}]

    def test_plain_dict_passthrough(self):
        d = {"a": 1, "b": [2, 3]}
        assert to_serializable(d) == {"a": 1, "b": [2, 3]}

    def test_plain_list_passthrough(self):
        assert to_serializable([1, 2, 3]) == [1, 2, 3]

    def test_primitives_passthrough(self):
        assert to_serializable(42) == 42
        assert to_serializable("hello") == "hello"
        assert to_serializable(None) is None

    def test_object_with_dict_public_attrs_only(self):
        class Simple:
            def __init__(self):
                self.x = 10
                self._private = 99

        result = to_serializable(Simple())
        assert result == {"x": 10}
        assert "_private" not in result

    def test_composable_with_sanitize(self):
        @dataclasses.dataclass
        class Metrics:
            accuracy: float
            loss: float

        plain = to_serializable(Metrics(0.95, float("nan")))
        clean = sanitize(plain)
        assert clean == {"accuracy": 0.95, "loss": None}


# ---------------------------------------------------------------------------
# save_html
# ---------------------------------------------------------------------------


class TestSaveHtml:
    def test_save_html_string(self, tmp_path):
        storage = LocalFileSystem(str(tmp_path))
        save_html("<h1>Hello</h1>", "report/index.html", storage)
        loaded = storage.load("report/index.html")
        assert loaded == b"<h1>Hello</h1>"

    def test_save_html_bytes(self, tmp_path):
        storage = LocalFileSystem(str(tmp_path))
        save_html(b"<p>raw bytes</p>", "report/raw.html", storage)
        loaded = storage.load("report/raw.html")
        assert loaded == b"<p>raw bytes</p>"

    def test_save_html_unicode(self, tmp_path):
        storage = LocalFileSystem(str(tmp_path))
        content = "<p>café</p>"
        save_html(content, "report/unicode.html", storage)
        loaded = storage.load("report/unicode.html")
        assert loaded == content.encode("utf-8")

    def test_save_html_nested_path(self, tmp_path):
        storage = LocalFileSystem(str(tmp_path))
        save_html("<html/>", "a/b/c/report.html", storage)
        loaded = storage.load("a/b/c/report.html")
        assert loaded == b"<html/>"


# ---------------------------------------------------------------------------
# save_plot
# ---------------------------------------------------------------------------


class TestSavePlot:
    def test_save_matplotlib_figure(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        storage = LocalFileSystem(str(tmp_path))
        save_plot(fig, "plots/chart.png", storage)
        plt.close(fig)

        loaded = storage.load("plots/chart.png")
        assert isinstance(loaded, bytes)
        # PNG magic bytes
        assert loaded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_plot_invalid_type_raises(self, tmp_path):
        storage = LocalFileSystem(str(tmp_path))
        with pytest.raises(TypeError, match="Unsupported figure type"):
            save_plot("not a figure", "plots/bad.png", storage)

    def test_save_plot_nested_path_created(self, tmp_path):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1])

        storage = LocalFileSystem(str(tmp_path))
        save_plot(fig, "deep/nested/path/chart.png", storage)
        plt.close(fig)

        loaded = storage.load("deep/nested/path/chart.png")
        assert isinstance(loaded, bytes)
        assert len(loaded) > 0


class TestHTMLReport:
    def test_basic_report(self):
        report = HTMLReport(title="Test", description="Desc", feature_name="churn")
        report.add_metric("accuracy", 0.95)
        report.add_metric("precision", 0.88)
        report.add_plot("loss", "report/loss.png")
        html = report.to_html()
        assert "<title>Test</title>" in html
        assert "churn" in html
        assert "0.95" in html
        assert "0.88" in html
        assert 'src="report/loss.png"' in html

    def test_sanitize_in_metrics(self):
        report = HTMLReport(title="X")
        report.add_metric("nan_val", float("nan"))
        html = report.to_html()
        assert "—" in html or "None" in html
