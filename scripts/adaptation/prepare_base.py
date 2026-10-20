"""Run official tensor conversion with float32 constructor to preserve master values."""
import importlib.util, shutil, sys
from common import *
def main():
 import torch
 assert (ASSETS/'official_base_verified.json').exists()
 dest=ASSETS/'pi0_base_torch'
 if (dest/'pretrained_provenance.json').exists():
  p=json.loads((dest/'pretrained_provenance.json').read_text());assert sha(dest/'model.safetensors')==p['weights_sha256'];return
 spec=importlib.util.spec_from_file_location('official_convert',ROOT/'third_party/openpi/examples/convert_jax_model_to_pytorch.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 tmp=dest.with_name(dest.name+'.partial')
 if tmp.exists():shutil.rmtree(tmp)
 torch.manual_seed(0)
 mod.convert_pi0_checkpoint(str(ASSETS/'pi0_base'),'float32',str(tmp),config().model)
 # The generic official assets are not LIBERO stats and must never enter this experiment.
 if (tmp/'assets').exists():shutil.rmtree(tmp/'assets')
 write(tmp/'pretrained_provenance.json',dict(pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_base',
  weights_sha256=sha(tmp/'model.safetensors'),official_inventory_sha256=sha(ASSETS/'base_inventory.json'),
  openpi_revision='981483dca0fd9acba698fea00aa6e52d56a66c58',conversion='official convert_pi0_checkpoint, Pi0Config(dtype=float32), precision=float32',seed=0))
 tmp.replace(dest)
 print('OFFICIAL_FLOAT32_BASE_CONVERTED',flush=True)
if __name__=='__main__':main()
