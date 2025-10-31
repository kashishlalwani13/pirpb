#!/usr/bin/env bash
set -euo pipefail
# Runs Stage C – Serper Search Retrieval

# Always run from repo root
cd "$(dirname "$0")/.."

# Activate venv
source .venv/bin/activate

# Ensure src is importable
export PYTHONPATH="$(pwd)/src"

# Run the module
python -m pirpb.search_serper

# Quick peek at output
echo ""
echo "Top of serp/serp_serper.csv:"
head -n 10 serp/serp_serper.csv || true
