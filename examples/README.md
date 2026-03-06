# mlplatform examples

Runnable examples for the two currently released sub-packages:
`mlplatform[utils]` and `mlplatform[config]`.

## 1. Install

### From PyPI / JFrog (once published)

```bash
pip install mlplatform[utils]    # serialisation + GCS upload helpers
pip install mlplatform[config]   # Pydantic pipeline config models
pip install mlplatform[core]     # both of the above in one command
```

### From this repository (local development)

```bash
# Install in editable mode so any source changes are reflected immediately
pip install -e "mlplatform/[core]"
```

---

## 2. Examples

| File | Sub-package | What it shows |
|---|---|---|
| `01_utils_serialization.py` | `mlplatform[utils]` | `sanitize` and `to_serializable` |
| `02_utils_storage.py` | `mlplatform[utils]` | `save_plot` and `save_html` with `LocalFileSystem` |
| `03_config_direct.py` | `mlplatform[config]` | Build `TrainingConfig` / `PredictionConfig` directly in Python |
| `04_config_from_yaml.py` | `mlplatform[config]` | Load a `PipelineConfig` from a YAML pipeline file |

Run any example from the **repository root**:

```bash
# Serialisation helpers
python examples/01_utils_serialization.py

# Storage helpers (writes to examples/output/ — safe to delete)
python examples/02_utils_storage.py

# Config: direct construction
python examples/03_config_direct.py

# Config: load from YAML
python examples/04_config_from_yaml.py
```

All examples print their output to stdout so you can see the results immediately.
Output artifacts (plots, HTML) are written to `examples/output/` and can be
deleted freely.
