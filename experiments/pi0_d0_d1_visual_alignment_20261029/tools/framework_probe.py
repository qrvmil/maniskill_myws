"""Run in separate fresh processes so JAX cannot retain VRAM during PyTorch smoke."""
from pathlib import Path
import json,sys
BASE=Path('/workspace/visual-alignment-repro-20261029')
from vram_gate import validate
validate(BASE/'evidence')
if sys.argv[1]=='jax':
 import jax,jax.numpy as jnp
 from jax._src.lib import cuda_versions
 assert jax.devices()[0].platform=='gpu'
 x=jnp.arange(1024*1024,dtype=jnp.float32).reshape(1024,1024)/1e6
 y=x@x.T;jax.block_until_ready(y)
 assert bool(jnp.isfinite(y).all())
 result=dict(version=jax.__version__,devices=[str(d) for d in jax.devices()],cuda_runtime=cuda_versions.cuda_runtime_get_version(),matmul_mean=float(y.mean()),passed=True)
else:
 import torch
 assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0)
 x=torch.arange(1024*1024,device='cuda',dtype=torch.float32).reshape(1024,1024)/1e6
 with torch.autocast('cuda',dtype=torch.bfloat16):y=x@x.T
 torch.cuda.synchronize();assert bool(torch.isfinite(y).all())
 result=dict(version=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),matmul_mean=float(y.mean()),passed=True)
(BASE/'evidence'/('framework-'+sys.argv[1]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
