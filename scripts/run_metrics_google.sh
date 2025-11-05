#!/usr/bin/env bash
# Stage D: run PIR metrics (Google CSE)

set -euo pipefail

# Move to repo root (this script lives in scripts/)
cd "$(dirname "$0")/.."

# Activate venv if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH=src

python -m pirpb.metrics

