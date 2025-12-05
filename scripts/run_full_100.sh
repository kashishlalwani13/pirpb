#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[INFO] Project root: $PROJECT_ROOT"

# Optional: activate virtualenv if it exists
if [ -d ".venv" ]; then
  echo "[INFO] Activating virtualenv .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found, continuing without activating a venv."
fi

echo
echo "========================================"
echo "[1/5] Paraphrasing all seed queries"
echo "========================================"
bash scripts/run_paraphraser.sh

echo
echo "========================================"
echo "[2/5] Retrieving SERPs via Google CSE"
echo "========================================"
bash scripts/run_google_cse.sh

echo
echo "========================================"
echo "[3/5] Computing item-level rank impact (item_pir)"
echo "========================================"
bash scripts/run_item_pir.sh

echo
echo "========================================"
echo "[4/5] Computing TRUE PIR + PB (pir_true_metric)"
echo "========================================"
python src/pirpb/pir_true_metric.py

echo
echo "========================================"
echo "[5/5] Computing bin-wise PIR and statistical tests"
echo "========================================"
python -m src.pirpb.pir_by_bin
python -m src.pirpb.stat_tests

echo
echo "[DONE] Full pipeline run for current base_queries.csv completed."
