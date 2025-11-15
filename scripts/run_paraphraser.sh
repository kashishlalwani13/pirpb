
set -euo pipefail
# Runs the Stage B paraphraser end-to-end using project config.
source .venv/bin/activate
python -m src.pirpb.paraphraser
