"""HTML report builder for ML experiment outputs."""

from __future__ import annotations

from typing import Any

from mlplatform.utils.serialization import sanitize


class HTMLReport:
    """Build HTML reports with embedded metrics and plots.

    Use :meth:`add_metric` and :meth:`add_plot` to populate the report,
    then call :meth:`to_html` to render. Save via
    :meth:`ArtifactRegistry.save` so plots are stored alongside the report.

    Example::

        report = HTMLReport(
            title="Model Report",
            description="Training results",
            feature_name="churn",
        )
        report.add_metric("accuracy", 0.95)
        report.add_metric("precision", precision_score(y, pred))
        report.add_plot("loss", "report/loss.png")  # path relative to artifact root
        artifact.save("report.html", report.to_html())
    """

    def __init__(
        self,
        title: str = "Model Report",
        description: str = "",
        feature_name: str | None = None,
    ) -> None:
        self.title = title
        self.description = description
        self.feature_name = feature_name or ""
        self._metrics: list[tuple[str, Any]] = []
        self._plots: list[tuple[str, str]] = []

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric. Value is sanitized for JSON/HTML safety."""
        self._metrics.append((name, sanitize(value)))

    def add_plot(self, name: str, path: str) -> None:
        """Add a plot reference. Path should be relative to the artifact root.

        Save the plot via ``artifact.save(path, fig)`` before building
        the report so it is stored in the same artifact directory.
        """
        self._plots.append((name, path))

    def to_html(self) -> str:
        """Render the report as HTML."""
        parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<meta charset='utf-8'>",
            f"<title>{_escape(self.title)}</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }",
            "h1 { color: #333; }",
            "table { border-collapse: collapse; margin: 1em 0; }",
            "th, td { border: 1px solid #ddd; padding: 0.5em 1em; text-align: left; }",
            "th { background: #f5f5f5; }",
            ".plot { margin: 1.5em 0; }",
            ".plot img { max-width: 100%; height: auto; }",
            ".plot figcaption { font-style: italic; color: #666; margin-top: 0.5em; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{_escape(self.title)}</h1>",
        ]
        if self.description:
            parts.append(f"<p>{_escape(self.description)}</p>")
        if self.feature_name:
            parts.append(f"<p><strong>Feature:</strong> {_escape(self.feature_name)}</p>")

        if self._metrics:
            parts.append("<h2>Metrics</h2>")
            parts.append("<table>")
            parts.append("<thead><tr><th>Metric</th><th>Value</th></tr></thead>")
            parts.append("<tbody>")
            for name, value in self._metrics:
                val_str = _format_value(value)
                parts.append(f"<tr><td>{_escape(name)}</td><td>{val_str}</td></tr>")
            parts.append("</tbody></table>")

        if self._plots:
            parts.append("<h2>Plots</h2>")
            for name, path in self._plots:
                parts.append('<div class="plot">')
                parts.append(f'<img src="{_escape(path)}" alt="{_escape(name)}">')
                parts.append(f'<figcaption>{_escape(name)}</figcaption>')
                parts.append("</div>")

        parts.append("</body></html>")
        return "\n".join(parts)


def _escape(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_value(value: Any) -> str:
    """Format a value for HTML display."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1e10 else str(value)
    return _escape(str(value))
