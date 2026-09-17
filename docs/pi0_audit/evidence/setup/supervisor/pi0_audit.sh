#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -eo pipefail
cd /workspace/maniskill_myws
export PYTHONPATH=/workspace/LIBERO:/workspace/maniskill_myws/src
export MUJOCO_GL=egl JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 TORCH_COMPILE_DISABLE=1
exec third_party/openpi/.venv/bin/python -u scripts/pld/audit_pi0.py eval --model positive_control
