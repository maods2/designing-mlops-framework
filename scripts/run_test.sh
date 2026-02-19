#!/usr/bin/env bash
# Run framework tests with virtual environment
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/test_framework.py
