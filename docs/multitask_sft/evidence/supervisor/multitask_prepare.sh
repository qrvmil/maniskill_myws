#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /root/maniskill_myws
export PYTHONPATH=/workspace/LIBERO:/root/maniskill_myws/src
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MUJOCO_GL=egl
pty third_party/openpi/.venv/bin/python -u scripts/prepare_multitask_sft.py fetch
pty third_party/openpi/.venv/bin/python -u scripts/prepare_multitask_sft.py prepare
