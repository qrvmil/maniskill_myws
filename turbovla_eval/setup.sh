#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/'; exit 1; }
uv venv .venv --python 3.10 --allow-existing
uv pip install --python .venv/bin/python torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt
if [ ! -d ../TurboVLA/.git ]; then git clone https://github.com/H-EmbodVis/TurboVLA.git ../TurboVLA; fi
git -C ../TurboVLA checkout a3365e8f1787d30bafd99d2806edbf4dc82b5370
if [ ! -d LIBERO/.git ]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO; fi
git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --python .venv/bin/python --no-deps -e ../TurboVLA -e LIBERO
.venv/bin/python download_assets.py
.venv/bin/python prepare_assets.py
.venv/bin/python prepare_observation.py
.venv/bin/python -m ipykernel install --user --name turbovla-libero --display-name 'TurboVLA LIBERO (Python 3.10)'
