#!/bin/bash
set -o pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/visual-alignment-repro-20261029/maniskill_myws
. experiments/pi0_d0_d1_visual_alignment_20261029/run-config.sh
exec third_party/openpi/.venv/bin/python -u experiments/pi0_d0_d1_visual_alignment_20261029/tools/run_stage.py
