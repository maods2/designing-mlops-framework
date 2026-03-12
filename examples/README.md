# mlplatform examples

Runnable examples for the released sub-packages:
`mlplatform[utils]`, `mlplatform[config]`, `mlplatform[artifacts]`.

## 1. Install

### From this repository (local development)

```bash
# Editable install
pip install -e ".[core]"

# Or from mlplatform package root:
pip install -e "mlplatform/[core]"
```

### For train/predict example (06)

```bash
pip install scikit-learn matplotlib
```

---

## 2. Examples

| File | What it shows |
|------|---------------|
| `01_utils_serialization.py` | `sanitize` and `to_serializable` for JSON-safe data |
| `03_config_direct.py` | `TrainingConfig` / `PredictionConfig` from kwargs dict or keyword args |
| `04_config_from_profiles.py` | `load_config_profiles` — load and merge YAML profiles into config |
| `05_artifacts_standalone.py` | `Artifact` — save/load with explicit params |
| `06_train_predict_workflow.py` | Train and predict with `TrainingConfig` and `Artifact` |

Run from the repository root:

```bash
python examples/01_utils_serialization.py
python examples/03_config_direct.py
python examples/04_config_from_profiles.py
python examples/05_artifacts_standalone.py
python examples/06_train_predict_workflow.py
```

Output artifacts are written to `examples/output/` and can be deleted freely.
