#!/bin/bash
set -euo pipefail
export HF_HOME=/workspace/.cache/huggingface
export PYTHONPATH=/workspace/pi0_experiment/src:/workspace/LIBERO
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
cd /workspace/pi0_experiment
while [ ! -f /workspace/pi0_three/runtime_ready ] || [ ! -f /workspace/pi0_three/assets/official_base_verified.json ]; do sleep 10; done
PYTHON=/workspace/pi0_venv/bin/python
"$PYTHON" -u scripts/adaptation/prepare_base.py
"$PYTHON" -u scripts/adaptation/prepare.py
"$PYTHON" -u scripts/adaptation/train_sft.py d0 --profile
# The registered microbatch is launched only after the profile completed successfully.
"$PYTHON" -u scripts/adaptation/train_sft.py d0
