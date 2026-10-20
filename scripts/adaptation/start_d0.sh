#!/bin/bash
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/.cache/huggingface
export PYTHONPATH=/workspace/pi0_experiment/src:/workspace/LIBERO
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
cd /workspace/pi0_experiment
while [ ! -f /workspace/pi0_three/runtime_ready ] || [ ! -f /workspace/pi0_three/assets/official_base_verified.json ]; do sleep 10; done
PYTHON=/workspace/pi0_venv/bin/python
if [ -f /workspace/pi0_three/evidence/d0_complete.json ]; then
  "$PYTHON" - <<'VERIFY'
import sys,json
sys.path.insert(0,'scripts/adaptation')
from common import WORK,NORM,sha
p=WORK/'sft/d0/3000'
m=json.loads((p/'training.json').read_text())
assert sha(p/'model.safetensors')==m['weights_sha256']
assert sha(NORM)==sha(p/'assets/d0/norm_stats.json')==m['normalization_sha256']
if not (p/'upload_receipt.json').exists():
 from publication_queue import submit
 submit(p,'checkpoints/d0/3000',True)
print('D0_ALREADY_COMPLETED_AND_VERIFIED')
VERIFY
  exit 0
fi
"$PYTHON" -m pytest -q tests/test_three_adaptations.py tests/test_pld_libero.py tests/test_pld_v3.py tests/test_pld_v4.py
"$PYTHON" -u scripts/adaptation/check_simulator.py
"$PYTHON" -u scripts/adaptation/prepare_base.py
"$PYTHON" -u scripts/adaptation/prepare.py
if [ ! -f /workspace/pi0_three/evidence/memory_profile.json ]; then
  "$PYTHON" -u scripts/adaptation/train_sft.py d0 --profile
else
  "$PYTHON" - <<'PROFILE'
import sys,json
sys.path.insert(0,'scripts/adaptation')
from common import WORK,MICROBATCH,BATCH
p=json.loads((WORK/'evidence/memory_profile.json').read_text())
assert p['effective_batch']==BATCH and p['microbatch']==MICROBATCH and p['fp32_master']
print('REUSING_VERIFIED_MEMORY_PROFILE')
PROFILE
fi
# The registered microbatch is launched only after the profile completed successfully.
"$PYTHON" -u scripts/adaptation/train_sft.py d0
