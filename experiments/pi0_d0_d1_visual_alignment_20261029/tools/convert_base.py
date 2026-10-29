"""Use the original official fp32 conversion, requiring the baseline input hash."""
from pathlib import Path
import importlib.util,json,shutil,sys,hashlib
import torch
from openpi.models.pi0_config import Pi0Config
BASE=Path('/workspace/visual-alignment-repro-20261029');ROOT=BASE/'maniskill_myws'
source=BASE/'private/official_base';dest=BASE/'private/pi0_base_torch'
assert (BASE/'evidence/official-base-receipts.json').exists()
assert not dest.exists(),'Never overwrite an existing converted checkpoint'
tmp=dest.with_name(dest.name+'.partial')
assert not tmp.exists(),'Retain prior interrupted conversion for investigation'
spec=importlib.util.spec_from_file_location('official_convert',ROOT/'third_party/openpi/examples/convert_jax_model_to_pytorch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(8);torch.manual_seed(0)
module.convert_pi0_checkpoint(str(source),'float32',str(tmp),Pi0Config(dtype='float32'))
if (tmp/'assets').exists():shutil.rmtree(tmp/'assets')
hash=hashlib.sha256()
with (tmp/'model.safetensors').open('rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):hash.update(block)
sha=hash.hexdigest();expected='8df3cbff6d06f55fc77202b7d66c671c5a9e05403620bd05c8eac95955f09926'
assert sha==expected,('Converted initial weights differ from historical official conversion',sha,expected)
result=dict(pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_base',conversion='official pinned converter, Pi0Config(dtype=float32), precision=float32,torch_seed0',weights_sha256=sha,historical_initial_weights_exact=True,openpi_revision='981483dca0fd9acba698fea00aa6e52d56a66c58')
(tmp/'pretrained_provenance.json').write_text(json.dumps(result,indent=2)+'\n');tmp.replace(dest)
(BASE/'evidence/base-conversion.json').write_text(json.dumps(result,indent=2)+'\n')
print('OFFICIAL_FP32_BASE_MATCHES_ORIGINAL',json.dumps(result),flush=True)
