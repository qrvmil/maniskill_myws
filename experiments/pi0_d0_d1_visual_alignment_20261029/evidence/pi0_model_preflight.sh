#!/bin/bash
set -o pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
cd /workspace/visual-alignment-repro-20261029/maniskill_myws
export HF_HOME=/workspace/.cache/huggingface
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/src"
python_bin="$PWD/third_party/openpi/.venv/bin/python"
preflight_tools="$PWD/experiments/pi0_d0_d1_visual_alignment_20261029/tools"
preflight_evidence=/workspace/visual-alignment-repro-20261029/evidence
printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" "$preflight_tools/vram_gate.py" "$preflight_evidence" || exit $?
/venv/main/bin/python -u "$preflight_tools/save_progress.py" "exp: save completed VRAM evidence before real model preflight" || exit $?
# Each framework GPU allocation is isolated to a short-lived process.
JAX_PLATFORMS=cuda "$python_bin" -u "$preflight_tools/framework_probe.py" jax > "$preflight_evidence/jax-gpu-probe.log" 2>&1
result=$?
if [ "$result" -eq 0 ]; then
  "$python_bin" -u "$preflight_tools/framework_probe.py" torch > "$preflight_evidence/torch-gpu-probe.log" 2>&1
  result=$?
fi
if [ "$result" -eq 0 ]; then
  "$python_bin" -u "$preflight_tools/model_smoke.py" > "$preflight_evidence/model-smoke.log" 2>&1
  result=$?
fi
printf '%s\n' "$result" > "$preflight_evidence/model-preflight.exitcode"
nvidia-smi -q > "$preflight_evidence/nvidia-smi-after-model.txt"
printf 'END_UTC=%s EXIT=%s\n' "$(date -u +%FT%TZ)" "$result"
/venv/main/bin/python -u "$preflight_tools/save_progress.py" "exp: preserve model preflight outcome before any long training"
publish_result=$?
if [ "$publish_result" -ne 0 ]; then exit "$publish_result"; fi
exit "$result"
