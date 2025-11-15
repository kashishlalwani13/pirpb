#!/usr/bin/env bash
set -euo pipefail
# Runs Stage C – Serper Search Retrieval
source .venv/bin/activate
PYTHONPATH=src python -m pirpb.search_serper
