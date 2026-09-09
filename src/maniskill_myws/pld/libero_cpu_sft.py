"""Full pi0 SFT using official forward/data transforms and CPU AdamW updates.

No additional parameter freezing, LoRA, quantization or action-head restriction.
Each step transfers the one model to GPU for forward/backward, then to CPU for
optimizer update. AdamW states always stay on CPU. Slow but bounded GPU storage.
"""
import json
from pathlib import Path
import shutil
import time


class ResidentCPUOptimizer:
    """One GPU model, CPU optimizer parameters/states; no second GPU model.

    Preserve the same AdamW math/dtypes while copying gradients down and updated
    parameters up. Avoid moving the entire module and its buffers each step.
    """
    def __init__(self,model,**optimizer_kwargs):
        import torch
        self.parameters=[torch.nn.Parameter(p.detach().to('cpu',copy=True),requires_grad=p.requires_grad)
                         for p in model.parameters()]
        self.optimizer=torch.optim.AdamW(self.parameters,**optimizer_kwargs)

    def step(self,model,*,grad_clip_norm=1):
        import torch
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip_norm) if grad_clip_norm else None
        if grad_norm is not None and not torch.isfinite(grad_norm):
            raise FloatingPointError(f'Nonfinite SFT gradients: {grad_norm}')
        for gpu,cpu in zip(model.parameters(),self.parameters,strict=True):
            cpu.grad=None if gpu.grad is None else gpu.grad.detach().to('cpu',copy=True)
        model.zero_grad(set_to_none=True)
        self.optimizer.step()
        with torch.no_grad():
            for gpu,cpu in zip(model.parameters(),self.parameters,strict=True):
                if cpu.grad is not None:gpu.copy_(cpu)
        self.optimizer.zero_grad(set_to_none=True)
        return None if grad_norm is None else float(grad_norm)


def cpu_optimizer_step(model, optimizer, *, grad_clip_norm=1):
    import torch
    # Same clipping as official PyTorch trainer; do it before moving gradients.
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip_norm) if grad_clip_norm else None
    if grad_norm is not None and not torch.isfinite(grad_norm):
        raise FloatingPointError(f'Nonfinite SFT gradients: {grad_norm}')
    parameters=list(model.parameters())
    model.to('cpu')
    if not all(p is q for p,q in zip(parameters,model.parameters(),strict=True)):
        raise RuntimeError('Moving model invalidated optimizer parameter references')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return None if grad_norm is None else float(grad_norm)


def train_cpu_offload(config,args,run):
    import jax
    import torch
    import safetensors.torch
    from .libero_openpi import load_pi0_pytorch
    from openpi.training.data_loader import create_data_loader
    from .libero_artifacts import write_json
    from .libero_protocol import file_sha256
    if not args.pytorch_base_checkpoint:
        raise ValueError('--pytorch-base-checkpoint is required for full CPU-offloaded SFT')
    base_root=Path(args.pytorch_base_checkpoint)
    provenance=json.loads((base_root/'pretrained_provenance.json').read_text())
    initial_hash=file_sha256(base_root/'model.safetensors')
    if provenance['pretrained_checkpoint']!='gs://openpi-assets/checkpoints/pi0_base' or provenance['weights_sha256']!=initial_hash:
        raise ValueError('Pretrained conversion provenance mismatch')
    norm_path=config.assets_dirs/args.repo_id/'norm_stats.json'
    if not norm_path.exists():raise ValueError('Source-only full-SFT normalization missing')
    binding={'initial_weights_sha256':initial_hash,'source_audit_sha256':file_sha256(Path(args.dataset_root)/'source_audit.json'),
             'normalization_sha256':file_sha256(norm_path),'seed':config.seed,'repo_id':args.repo_id,
             'batch_size':config.batch_size,'lr_schedule':repr(config.lr_schedule),'optimizer':repr(config.optimizer)}
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config.seed)
    import numpy as np
    np.random.seed(config.seed)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0,16*1024**3/total))
    start_step=0
    weight_root=Path(args.resume_checkpoint or args.pytorch_base_checkpoint)
    model=load_pi0_pytorch(config.model,weight_root/'model.safetensors')
    model.gradient_checkpointing_enable();model.train()
    optimizer_kwargs=dict(lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1,config.optimizer.b2),eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,foreach=False)
    resident=ResidentCPUOptimizer(model,**optimizer_kwargs) if args.optimizer_storage=='resident_cpu' else None
    optimizer=resident.optimizer if resident else torch.optim.AdamW(model.parameters(),**optimizer_kwargs)
    if args.resume_checkpoint:
        saved=torch.load(weight_root.parent/'resume_state.pt',map_location='cpu',weights_only=False)
        if saved['checkpoint']!=str(weight_root.resolve()):
            raise ValueError('Resume optimizer belongs to a different checkpoint')
        if saved['weights_sha256']!=file_sha256(weight_root/'model.safetensors'):
            raise ValueError('Resume weights checksum mismatch')
        if saved['binding']!=binding:
            raise ValueError('Resume training data/configuration mismatch')
        optimizer.load_state_dict(saved['optimizer']);start_step=saved['step']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    loader=create_data_loader(config,framework='pytorch',shuffle=True)
    iterator=iter(loader)
    # Reproduce shuffled dataloader position for a restarted process. Loader
    # iteration/CPU decoding does not consume training's GPU RNG.
    for _ in range(start_step):next(iterator)
    run.meta.update(full_model=True,optimizer_device='cpu',batch_size=config.batch_size,
        optimizer_storage=args.optimizer_storage,
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_weights_sha256=file_sha256(Path(args.pytorch_base_checkpoint)/'model.safetensors'))
    checkpoint=None
    for step in range(start_step,config.num_train_steps):
        started=time.perf_counter()
        observation,actions=next(iterator)
        loaded=time.perf_counter()
        model.to('cuda')
        observation=jax.tree.map(lambda x:x.to('cuda'),observation)
        actions=actions.to(device='cuda',dtype=torch.float32)
        torch.cuda.synchronize()
        transferred=time.perf_counter()
        schedule=config.lr_schedule
        # Matches the official train_pytorch.py schedule.
        if step<schedule.warmup_steps:
            lr=schedule.peak_lr*(step+1)/(schedule.warmup_steps+1)
        else:
            import math
            progress=min(1.,(step-schedule.warmup_steps)/max(1,schedule.decay_steps-schedule.warmup_steps))
            lr=schedule.decay_lr+.5*(schedule.peak_lr-schedule.decay_lr)*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=lr
        loss=model(observation,actions).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SFT loss')
        value=float(loss.detach());loss.backward()
        torch.cuda.synchronize()
        backward_done=time.perf_counter()
        grad_norm=(resident.step(model,grad_clip_norm=config.optimizer.clip_gradient_norm) if resident else
                   cpu_optimizer_step(model,optimizer,grad_clip_norm=config.optimizer.clip_gradient_norm))
        del loss,observation,actions
        torch.cuda.synchronize()
        metrics={'step':step+1,'loss':value,'grad_norm':grad_norm,'learning_rate':lr,
                 'seconds':time.perf_counter()-started,
                 'data_seconds':loaded-started,'transfer_seconds':transferred-loaded,
                 'forward_backward_seconds':backward_done-transferred,
                 'cpu_offload_optimizer_seconds':time.perf_counter()-backward_done,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        with (run.path/'logs/sft.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if (step+1)%config.save_interval==0 or step+1==config.num_train_steps:
            checkpoint=run.path/'checkpoints'/str(step+1)
            checkpoint.mkdir(exist_ok=False)
            safetensors.torch.save_model(model,str(checkpoint/'model.safetensors'))
            target=checkpoint/'assets'/args.repo_id;target.mkdir(parents=True)
            shutil.copy2(norm_path,target/'norm_stats.json')
            write_json(checkpoint/'training.json',{'step':step+1,'source_only':True,
                'method':'full_cpu','config':repr(config),'initial_weights_sha256':run.meta['initial_weights_sha256']})
            state={'step':step+1,'checkpoint':str(checkpoint.resolve()),'optimizer':optimizer.state_dict(),
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'weights_sha256':file_sha256(checkpoint/'model.safetensors'),'binding':binding}
            temp=run.path/'checkpoints/resume_state.tmp'
            torch.save(state,temp);temp.replace(run.path/'checkpoints/resume_state.pt')
            run.meta['checkpoint']=str(checkpoint)
            write_json(run.path/'progress.json',dict(step=step+1,checkpoint=str(checkpoint)))
    if checkpoint is None:raise ValueError('No new steps requested')
    return checkpoint
