#!/bin/bash
set -o pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/visual-alignment-repro-20261029/maniskill_myws
. experiments/pi0_d0_d1_visual_alignment_20261029/run-config.sh
preflight_tools="$PWD/experiments/pi0_d0_d1_visual_alignment_20261029/tools"
/venv/main/bin/python -u "$preflight_tools/save_progress.py" "exp: save preflight and alignment cache implementation before D0 cache" || exit $?
third_party/openpi/.venv/bin/python -u "$preflight_tools/prepare_alignment.py" cache D0 2>&1 | tee /workspace/visual-alignment-repro-20261029/evidence/alignment-cache-D0.log
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > /workspace/visual-alignment-repro-20261029/evidence/alignment-cache-D0.exitcode
/venv/main/bin/python -u "$preflight_tools/save_progress.py" "exp: preserve D0 C-RADIO cache outcome"
exit "$result"
