#!/usr/bin/env python3
"""Actual pi0 full-model training allocation probe; synthetic input, no SFT claim."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('JAX_PLATFORMS','cpu')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maniskill_myws.pld.libero_artifacts import RunArtifacts,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--budget-gib',type=float,default=16)
    p.add_argument('--cpu-optimizer',action='store_true')
    args=p.parse_args()
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1,args.budget_gib*1024**3/total))
    with RunArtifacts(args.output,vars(args)) as run:
        model=PI0Pytorch(Pi0Config()).to('cuda')
        model.gradient_checkpointing_enable()
        model.train()
        counts={}
        for param in model.parameters():
            k=str(param.dtype);counts[k]=counts.get(k,0)+param.numel()
        run.meta.update(parameter_counts=counts,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                        alignment=False,input='synthetic; actual full-size official pi0 architecture',batch_size=1)
        write_json(run.path/'parameter_inventory.json',run.meta)
        obs=Observation(images={k:torch.zeros(1,3,224,224,device='cuda') for k in ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']},
            image_masks={k:torch.tensor([k!='right_wrist_0_rgb'],device='cuda') for k in ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']},
            state=torch.zeros(1,32,device='cuda'),tokenized_prompt=torch.ones(1,48,dtype=torch.long,device='cuda'),
            tokenized_prompt_mask=torch.ones(1,48,dtype=torch.bool,device='cuda'))
        opt=torch.optim.AdamW(model.parameters(),lr=2.5e-5,foreach=False)
        run.meta['phase']='forward'
        loss=model(obs,torch.zeros(1,50,32,device='cuda')).mean()
        run.meta.update(phase='backward',loss=float(loss.detach()))
        loss.backward()
        run.meta['phase']='optimizer'
        if args.cpu_optimizer:
            original=list(model.parameters())
            model.to('cpu')
            assert all(p is q for p,q in zip(original,model.parameters()))
        opt.step()
        run.meta['phase']='completed_update'
        torch.cuda.synchronize()

if __name__=='__main__':
    main()
