#!/usr/bin/env bash
set -euo pipefail

# Always run from project root (scripts/..)
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "❌ .venv not found. Create and install dependencies first."
  exit 1
fi

# Ensure package imports work
export PYTHONPATH=src

# Run item-level PIR module
python -m pirpb.item_pir
