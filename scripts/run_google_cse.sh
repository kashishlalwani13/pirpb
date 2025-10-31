#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src"
python -m pirpb.search_google_cse
echo ""
echo "Top of serp/serp_google.csv:"
head -n 10 serp/serp_google.csv || true
