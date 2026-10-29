#!/bin/bash
set -o pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/visual-alignment-repro-20261029/evidence
printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.free,temperature.gpu,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
/workspace/visual-alignment-repro-20261029/cuda_memtest/build/cuda_memtest --device 0 --num_passes 1 --num_iterations 1000 --exit_on_error > cuda_memtest.log 2>&1
result=$?
printf '%s\n' "$result" > cuda_memtest.exitcode
nvidia-smi -q > nvidia-smi-after.txt
printf 'END_UTC=%s EXIT=%s\n' "$(date -u +%FT%TZ)" "$result"
exit "$result"
