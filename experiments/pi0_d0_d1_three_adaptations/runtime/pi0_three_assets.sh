#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export HF_HOME=/workspace/.cache/huggingface
cd /workspace/pi0_experiment
exec /venv/main/bin/python -u scripts/adaptation/download_assets.py
