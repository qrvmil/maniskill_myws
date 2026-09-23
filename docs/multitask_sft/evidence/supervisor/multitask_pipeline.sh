#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /root/maniskill_myws
export PYTHONPATH=/workspace/LIBERO:/root/maniskill_myws/src
export HF_LEROBOT_HOME=/workspace/multitask-sft/lerobot
export MUJOCO_GL=egl JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 TORCH_COMPILE_DISABLE=1
pty third_party/openpi/.venv/bin/python -u scripts/run_multitask_sft.py
