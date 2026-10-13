#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${TURBOVLA_ENV:-/workspace/turbovla-venv}
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
uv venv "$ENV_DIR" --python 3.10 --allow-existing
uv pip install --python "$ENV_DIR/bin/python" torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$ENV_DIR/bin/python" -r "$ROOT/turbovla_eval/requirements.txt" tensorflow-cpu==2.20.0 tensorflow-datasets==4.9.3
test -d "$ROOT/TurboVLA/.git" || git clone https://github.com/H-EmbodVis/TurboVLA.git "$ROOT/TurboVLA"
git -C "$ROOT/TurboVLA" checkout a3365e8f1787d30bafd99d2806edbf4dc82b5370
test -d "$ROOT/turbovla_eval/LIBERO/.git" || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$ROOT/turbovla_eval/LIBERO"
git -C "$ROOT/turbovla_eval/LIBERO" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --python "$ENV_DIR/bin/python" --no-deps -e "$ROOT/TurboVLA" -e "$ROOT/turbovla_eval/LIBERO"
"$ENV_DIR/bin/python" -m ipykernel install --user --name turbovla-multitask --display-name 'TurboVLA multitask (Python 3.10)'
