#!/bin/bash
# ===========================================
# Stage E — Perturbation Bias (PB)
# ===========================================
# Computes vendor-domain bias using serp/serp_google.csv,
# paraphrases, and domain_map.yaml.

set -e
export PYTHONPATH=src

echo "[Stage E] Starting Perturbation Bias (PB) Analysis ..."
python -m pirpb.pb
echo "[Stage E] PB computation complete."
