#!/usr/bin/env python3
"""Download only official pretrained pi0 and convert with pinned OpenPI code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_protocol import directory_manifest,file_sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--verify-existing',action='store_true',help='Verify existing conversion against a fresh official conversion')
    args=p.parse_args()
    from openpi.shared.download import maybe_download
    source=Path(maybe_download('gs://openpi-assets/checkpoints/pi0_base'))
    output=Path(args.output).resolve()
    destination=output.with_name(output.name+'-verification') if args.verify_existing else output
    if destination.exists():raise FileExistsError(destination)
    command=[sys.executable,str(ROOT/'third_party/openpi/examples/convert_jax_model_to_pytorch.py'),
             '--checkpoint-dir',str(source),'--config-name','pi0_libero',
             '--output-path',str(destination),'--precision','bfloat16']
    subprocess.run(command,check=True)
    weights_hash=file_sha256(destination/'model.safetensors')
    ignored_random=[]
    if args.verify_existing and weights_hash!=file_sha256(output/'model.safetensors'):
        # Upstream instantiates an unused expert lm_head absent from pi0 JAX.
        # Verify every functional tensor, not nondeterministic container bytes.
        from safetensors import safe_open
        import torch
        with safe_open(str(output/'model.safetensors'),framework='pt') as a,safe_open(str(destination/'model.safetensors'),framework='pt') as b:
            if set(a.keys())!=set(b.keys()):raise ValueError('Conversion key mismatch')
            for key in a.keys():
                if not torch.equal(a.get_tensor(key),b.get_tensor(key)):
                    if key!='paligemma_with_expert.gemma_expert.lm_head.weight':
                        raise ValueError(f'Functional conversion tensor differs: {key}')
                    ignored_random.append(key)
        weights_hash=file_sha256(output/'model.safetensors')
    manifest={'pretrained_checkpoint':'gs://openpi-assets/checkpoints/pi0_base',
              'weights_sha256':weights_hash,'source_files':directory_manifest(source),
              'conversion_command':command,'ignored_unused_random_parameters':ignored_random,'openpi_revision':subprocess.check_output(
                  ['git','-C',str(ROOT/'third_party/openpi'),'rev-parse','HEAD'],text=True).strip()}
    (output/'pretrained_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
