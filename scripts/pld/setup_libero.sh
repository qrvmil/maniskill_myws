#!/usr/bin/env bash
# Pinned simulator + OpenPI environment; framework wheel supports RTX 5080.
set -euo pipefail
PLD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PLD_ROOT"
git submodule update --init third_party/openpi
PLD_LIBERO_ROOT=${PLD_LIBERO_ROOT:-/workspace/LIBERO}
if [[ ! -d "$PLD_LIBERO_ROOT/.git" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$PLD_LIBERO_ROOT"
fi
if [[ -n $(git -C "$PLD_LIBERO_ROOT" status --porcelain) ]]; then
    echo 'LIBERO checkout is dirty; preserve it and provide a clean PLD_LIBERO_ROOT.' >&2
    exit 1
fi
git -C "$PLD_LIBERO_ROOT" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
(
    cd third_party/openpi
    GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
)
uv pip install --python third_party/openpi/.venv/bin/python \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python third_party/openpi/.venv/bin/python \
    'numpy==1.26.4' 'robosuite==1.4.1' 'mujoco==3.2.7' 'bddl==1.0.1' \
    'gym==0.26.2' future pytest h5py easydict
cp -r third_party/openpi/src/openpi/models_pytorch/transformers_replace/. \
    third_party/openpi/.venv/lib/python3.11/site-packages/transformers/
export PLD_LIBERO_ROOT
third_party/openpi/.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml
root=Path(os.environ['PLD_LIBERO_ROOT'])/'libero/libero'
target=Path.home()/'.libero/config.yaml'
data={k:str(root/v) for k,v in [('benchmark_root',''),('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]}
data['datasets']=str(root.parent/'datasets')
if target.exists() and yaml.safe_load(target.read_text())!=data:
    raise RuntimeError(f'Existing {target} differs; preserve it and resolve paths explicitly')
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(yaml.safe_dump(data))
PY
scripts/pld/libero_python.sh -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name()); print((torch.ones(1,device="cuda")+1).item())'
