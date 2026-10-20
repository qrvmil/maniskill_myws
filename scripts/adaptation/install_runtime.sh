#!/bin/bash
set -euo pipefail
export HF_HOME=/workspace/.cache/huggingface
uv pip install --python /venv/main/bin/python huggingface_hub==2.2.0 hf_xet==1.7.0 numpy==1.26.4 requests==2.32.3
uv pip install --python /workspace/pi0_venv/bin/python 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /workspace/pi0_venv/bin/python 'future==0.18.2' 'robosuite==1.4.1' 'bddl==1.0.1' 'easydict==1.9' 'gym==0.25.2' 'mujoco==3.2.7' matplotlib nbformat nbclient ipykernel pytest
uv pip install --python /workspace/pi0_venv/bin/python --no-deps -e /workspace/LIBERO
cp -r /workspace/pi0_experiment/third_party/openpi/src/openpi/models_pytorch/transformers_replace/* /workspace/pi0_venv/lib/python3.11/site-packages/transformers/
mkdir -p /root/.libero
/workspace/pi0_venv/bin/python - <<'PY'
import pathlib,yaml
r=pathlib.Path('/workspace/LIBERO/libero/libero')
d=dict(benchmark_root=str(r),bddl_files=str(r/'bddl_files'),init_states=str(r/'init_files'),datasets='/workspace/pi0_three/assets/native',assets=str(r/'assets'))
pathlib.Path('/root/.libero/config.yaml').write_text(yaml.safe_dump(d))
import torch
print('TORCH_RUNTIME',torch.__version__,torch.version.cuda,torch.cuda.get_device_name())
a=torch.randn((1024,1024),device='cuda'); print('BLACKWELL_MATMUL',float((a@a.T).mean()))
PY
touch /workspace/pi0_three/runtime_ready
