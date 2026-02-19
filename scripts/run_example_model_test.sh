#!/usr/bin/env bash
# Run example_model tests with virtual environment
# Requires Python 3.9+ (for type hints) and: pip install pyyaml
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/mlplatform"
python3 scripts/test_example_model.py
