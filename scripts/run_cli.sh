#!/usr/bin/env bash
# Run mlplatform CLI with monorepo root on PYTHONPATH
# Usage: ./scripts/run_cli.sh run --project-root template_model --train-data template_model/data/sample_train.csv ...
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
source .venv/bin/activate
exec mlplatform "$@"
