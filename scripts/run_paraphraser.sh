#!/usr/bin/env bash
set -euo pipefail
# Runs the Stage B paraphraser end-to-end using project config.
source .venv/bin/activate
PYTHONPATH=src python -m pirpb.paraphraser
