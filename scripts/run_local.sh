#!/bin/bash
#
# Run the full local workflow: training, in-process prediction, and PySpark batch prediction.
#
# Usage:
#   cd /workspaces/ml-platform-v2
#   bash scripts/run_local.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/mlplatform"

VERSION="local_$(date +%Y%m%d_%H%M%S)"
BASE_PATH="${REPO_ROOT}/artifacts_local_run"
DIST_DIR="${REPO_ROOT}/dist_local_run"

TRAIN_DAG="${REPO_ROOT}/example_model/pipeline/train.yaml"
PREDICT_DAG="${REPO_ROOT}/example_model/pipeline/predict.yaml"

cleanup() {
    echo ""
    echo "Cleaning up..."
    rm -rf "${BASE_PATH}" "${DIST_DIR}"
    echo "Removed ${BASE_PATH} and ${DIST_DIR}"
}

echo "============================================"
echo " mlplatform - Local Workflow"
echo "============================================"
echo "  REPO_ROOT  : ${REPO_ROOT}"
echo "  VERSION    : ${VERSION}"
echo "  BASE_PATH  : ${BASE_PATH}"
echo "  TRAIN_DAG  : ${TRAIN_DAG}"
echo "  PREDICT_DAG: ${PREDICT_DAG}"
echo "============================================"
echo ""

# -----------------------------------------------
# Step 1: Training via Python API
# -----------------------------------------------
echo ">>> Step 1: Training (Python API)"
echo "    Loading DAG, generating synthetic data, running trainer..."

python3 -c "
import pandas as pd
from sklearn.datasets import make_classification
from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.runner import _build_context, _run_training

pipeline = ConfigLoaderFactory.load_pipeline_config('${TRAIN_DAG}', task_id='train_model', config_names=['global', 'train-local'])
task_cfg = pipeline.tasks[0]
model_cfg = task_cfg.to_model_config()
ctx = _build_context(pipeline, task_cfg, 'local', '${VERSION}', '${BASE_PATH}')

X, y = make_classification(n_samples=100, n_features=5, random_state=42)
ctx.optional_configs['train_data'] = {
    'X': pd.DataFrame(X, columns=['f0','f1','f2','f3','f4']),
    'y': pd.Series(y),
}

_run_training(model_cfg, ctx)
print('    Training complete.')
"

echo ""

# -----------------------------------------------
# Step 2: In-process prediction via Python API
# -----------------------------------------------
echo ">>> Step 2: In-process Prediction (Python API)"
echo "    Loading trained model, running predict_chunk on sample data..."

python3 -c "
import pandas as pd
from sklearn.datasets import make_classification
from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.runner import _build_context

pipeline = ConfigLoaderFactory.load_pipeline_config('${PREDICT_DAG}', task_id='predict', config_names=['global', 'predict-local'])
task_cfg = pipeline.tasks[0]
ctx = _build_context(pipeline, task_cfg, 'local', '${VERSION}', '${BASE_PATH}')

from example_model.predict import MyPredictor
predictor = MyPredictor()
predictor.context = ctx
predictor.load_model()

X_test, _ = make_classification(n_samples=5, n_features=5, random_state=99)
test_df = pd.DataFrame(X_test, columns=['f0','f1','f2','f3','f4'])
result = predictor.predict(test_df)

print('    Predictions:')
print(result[['f0','prediction']].to_string(index=False))
print(f'    Rows: {len(result)}, Columns: {list(result.columns)}')
print('    In-process prediction complete.')
"

echo ""

# -----------------------------------------------
# Step 3: PySpark batch prediction
# -----------------------------------------------
echo ">>> Step 3: PySpark Batch Prediction"
echo "    Serializing config, creating input CSV, running spark/main.py..."

mkdir -p "${DIST_DIR}"

# 3a: Serialize prediction config JSON
python3 -c "
from mlplatform.config.factory import ConfigLoaderFactory
from mlplatform.spark.config_serializer import write_workflow_config

pipeline = ConfigLoaderFactory.load_pipeline_config('${PREDICT_DAG}', task_id='predict', config_names=['global', 'predict-local'])
task_cfg = pipeline.tasks[0]
write_workflow_config(pipeline, task_cfg, '${DIST_DIR}/spark_config.json',
                      base_path='${BASE_PATH}', version='${VERSION}')
print('    Config written to ${DIST_DIR}/spark_config.json')
"

# 3b: Create input CSV
python3 -c "
import pandas as pd
from sklearn.datasets import make_classification

X, _ = make_classification(n_samples=20, n_features=5, random_state=77)
df = pd.DataFrame(X, columns=['f0','f1','f2','f3','f4'])
df.to_csv('${DIST_DIR}/input.csv', index=False)
print('    Input CSV written (20 rows)')
"

# 3c: Run Spark prediction
python3 "${REPO_ROOT}/mlplatform/mlplatform/spark/main.py" \
    --config "${DIST_DIR}/spark_config.json" \
    --input-path "${DIST_DIR}/input.csv" \
    --output-path "${DIST_DIR}/predictions.parquet"

# 3d: Verify output
python3 -c "
import pandas as pd
result = pd.read_parquet('${DIST_DIR}/predictions.parquet')
print(f'    Output: {len(result)} rows, columns: {list(result.columns)}')
print(result[['f0','prediction']].head(5).to_string(index=False))
print('    PySpark batch prediction complete.')
"

echo ""

# -----------------------------------------------
# Step 4: Inspect artifacts
# -----------------------------------------------
echo ">>> Step 4: Artifact Inspection"

python3 -c "
import json
from pathlib import Path

base = Path('${BASE_PATH}')
registry = base / 'model_registry.json'
if registry.exists():
    with open(registry) as f:
        data = json.load(f)
    for model, entries in data.items():
        for entry in entries:
            print(f'    Model: {model}, Version: {entry.get(\"version\")}, Accuracy: {entry.get(\"accuracy\", \"N/A\")}')
            print(f'    Path:  {base / entry.get(\"path\", \"\")}')

artifacts = list(base.rglob('*.pkl'))
print(f'    Total artifacts: {len(artifacts)} .pkl files')
for a in artifacts:
    print(f'      {a.relative_to(base)}')
"

echo ""
echo "============================================"
echo " All steps completed successfully."
echo "============================================"

cleanup
