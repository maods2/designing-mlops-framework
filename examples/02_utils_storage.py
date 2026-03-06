"""Example: mlplatform.utils — save_plot and save_html.

Demonstrates saving a matplotlib figure and an HTML report via the
Storage interface (LocalFileSystem).  Swap LocalFileSystem for GCSStorage
and no other code changes are needed.

Install
-------
    pip install mlplatform[utils]
    # or, from this repo:
    pip install -e "mlplatform/[utils]"

Run
---
    python examples/02_utils_storage.py

Output is written to examples/output/ — safe to delete.
"""

import sys
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

import matplotlib.pyplot as plt  # noqa: E402

from mlplatform.storage.local import LocalFileSystem  # noqa: E402
from mlplatform.utils import save_html, save_plot  # noqa: E402

# All artifacts for this example go under examples/output/
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
storage = LocalFileSystem(str(OUTPUT_DIR))


# ── 1. save_plot — persist a matplotlib figure ──────────────────────────────

print("=" * 60)
print("1. save_plot — serialize a matplotlib figure to PNG via storage")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: training/validation loss curves
epochs = list(range(1, 11))
train_loss = [0.9, 0.7, 0.55, 0.44, 0.36, 0.30, 0.26, 0.23, 0.21, 0.19]
val_loss   = [0.95, 0.75, 0.62, 0.52, 0.46, 0.42, 0.40, 0.39, 0.38, 0.38]

axes[0].plot(epochs, train_loss, label="train")
axes[0].plot(epochs, val_loss,   label="validation", linestyle="--")
axes[0].set_title("Loss curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Right: precision / recall / F1 bar chart
metrics_names  = ["Precision", "Recall", "F1"]
metrics_values = [0.93, 0.91, 0.92]
axes[1].bar(metrics_names, metrics_values, color=["steelblue", "tomato", "seagreen"])
axes[1].set_ylim(0, 1)
axes[1].set_title("Evaluation metrics")
axes[1].set_ylabel("Score")

fig.suptitle("Model training report", fontsize=14)
fig.tight_layout()

plot_path = "reports/training_report.png"
save_plot(fig, plot_path, storage)
plt.close(fig)

saved_bytes = storage.load(plot_path)
print(f"\nSaved PNG  → {OUTPUT_DIR / plot_path}")
print(f"Size: {len(saved_bytes):,} bytes")


# ── 2. save_html — persist an HTML report ────────────────────────────────────

print("\n" + "=" * 60)
print("2. save_html — persist an HTML report via storage")
print("=" * 60)

html_report = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Evaluation Report</title>
  <style>
    body  { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; }
    table { border-collapse: collapse; width: 100%; }
    th, td{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th    { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>Model Evaluation Report</h1>
  <p>Pipeline: <strong>churn_model / v1.0</strong></p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Accuracy</td> <td>0.9500</td></tr>
    <tr><td>Precision</td><td>0.9300</td></tr>
    <tr><td>Recall</td>   <td>0.9100</td></tr>
    <tr><td>F1 Score</td> <td>0.9200</td></tr>
  </table>
</body>
</html>"""

html_path = "reports/evaluation_report.html"
save_html(html_report, html_path, storage)

saved_html_bytes = storage.load(html_path)
print(f"\nSaved HTML → {OUTPUT_DIR / html_path}")
print(f"Size: {len(saved_html_bytes):,} bytes")


# ── 3. Round-trip: reload and inspect ────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Round-trip — reload artifacts from storage")
print("=" * 60)

png_bytes  = storage.load(plot_path)
html_bytes = storage.load(html_path)

print(f"\nReloaded PNG  : {len(png_bytes):,} bytes  (starts with {png_bytes[:4]!r})")
print(f"Reloaded HTML : {len(html_bytes):,} bytes  (starts with {html_bytes[:15]!r}...)")

# Tip: swap LocalFileSystem for GCSStorage — nothing else changes:
#
#   from mlplatform.storage.gcs import GCSStorage
#   storage = GCSStorage("gs://my-bucket/artifacts")
#   save_plot(fig, "reports/training_report.png", storage)
#   png_bytes = storage.load("reports/training_report.png")

print("\nDone. Output written to:", OUTPUT_DIR)
