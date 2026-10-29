#!/bin/bash
set -o pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/visual-alignment-repro-20261029/evidence
printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)"
nvidia-smi -q > nvidia-smi-before-stress64.txt
/workspace/visual-alignment-repro-20261029/cuda_memtest/build/cuda_memtest --device 0 --stress --num_iterations 1000 --num_passes 1 > vram-stress64.log 2>&1
result=$?
printf '%s\n' "$result" > vram-stress64.exitcode
nvidia-smi -q > nvidia-smi-after-stress64.txt
printf 'END_UTC=%s EXIT=%s\n' "$(date -u +%FT%TZ)" "$result"
exit "$result"
