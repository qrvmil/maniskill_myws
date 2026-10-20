#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export HF_HOME=/workspace/.cache/huggingface
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
cd /workspace/pi0_experiment
exec /venv/main/bin/python -u scripts/adaptation/publication_queue.py serve
