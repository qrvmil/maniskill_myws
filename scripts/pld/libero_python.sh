#!/usr/bin/env bash
set -euo pipefail
PLD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PLD_LIBERO_ROOT:-/workspace/LIBERO}:${PLD_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export JAX_PLATFORMS=cpu
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=4
cd "$PLD_ROOT"
exec third_party/openpi/.venv/bin/python "$@"
