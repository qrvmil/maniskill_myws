#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export HF_HOME=/workspace/.cache/huggingface
export PYTHONPATH=/workspace/pi0_experiment/src:/workspace/LIBERO
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
cd /workspace/pi0_experiment
while [ ! -f /workspace/pi0_three/runtime_ready ]; do sleep 5; done
exec /workspace/pi0_venv/bin/python -u scripts/adaptation/pipeline.py
