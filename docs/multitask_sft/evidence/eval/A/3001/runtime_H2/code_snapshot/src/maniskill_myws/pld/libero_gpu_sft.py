"""Source-audited wrapper around pinned OpenPI's official GPU PyTorch trainer."""
import dataclasses
import json
import logging
from pathlib import Path


def retain_latest_optimizer(root,latest_step):
    """Keep all policy weights and only the latest resumable optimizer state."""
    root=Path(root)
    if not all((root/str(latest_step)/name).is_file() for name in ('model.safetensors','optimizer.pt')):
        raise ValueError('Latest checkpoint must be complete before optimizer retention')
    removed=[]
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.isdecimal() and int(path.name)<latest_step:
            optimizer=path/'optimizer.pt'
            if optimizer.is_file():
                optimizer.unlink();removed.append(str(optimizer.relative_to(root)))
    return removed


def save_completed_update(save_fn,model,optimizer,step,config,is_main,data_config):
    """Fix upstream's N-1 final-save check after its counter was incremented.

    Keep native serialization. Only registered interval/final saves are allowed;
    force the final save through the native interval branch when necessary.
    """
    if step>0 and (step%config.save_interval==0 or step==config.num_train_steps):
        save_config=(dataclasses.replace(config,save_interval=step)
                     if step==config.num_train_steps else config)
        save_fn(model,optimizer,step,save_config,is_main,data_config)


def train_official_pytorch(config,args,run,trainer):
    import torch
    from .libero_protocol import file_sha256
    from .libero_artifacts import write_json
    if args.resume_checkpoint:
        raise ValueError('Official GPU path starts from pretrained weights; CPU-offload resume is separate')
    if not args.pytorch_base_checkpoint:
        raise ValueError('Official pretrained PyTorch conversion is required')
    root=Path(args.pytorch_base_checkpoint)
    provenance=json.loads((root/'pretrained_provenance.json').read_text())
    weight_hash=file_sha256(root/'model.safetensors')
    if (provenance['pretrained_checkpoint']!='gs://openpi-assets/checkpoints/pi0_base'
            or provenance['weights_sha256']!=weight_hash):
        raise ValueError('Pretrained conversion provenance mismatch')
    torch.set_num_threads(args.cpu_threads)
    config=dataclasses.replace(config,pytorch_weight_path=str(root.resolve()),
        checkpoint_base_dir=str((run.path/'checkpoints').resolve()),log_interval=1,
        overwrite=False,resume=False)
    (run.path/'official_gpu_config.txt').write_text(repr(config))
    trainer.init_logging()
    handler=logging.FileHandler(run.path/'logs/sft.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    original_save=trainer.save_checkpoint
    def save_with_audit(model,optimizer,step,cfg,is_main,data):
        if step==1:
            run.meta.update(parameter_count=sum(p.numel() for p in model.parameters()),
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        save_completed_update(original_save,model,optimizer,step,cfg,is_main,data)
        if is_main and (step%cfg.save_interval==0 or step==cfg.num_train_steps):
            removed=retain_latest_optimizer(cfg.checkpoint_dir,step)
            with (run.path/'logs/checkpoint_retention.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,removed_optimizer_files=removed,
                                       all_policy_weights_retained=True))+'\n')
    trainer.save_checkpoint=save_with_audit
    run.meta.update(full_model=True,optimizer_device='cuda',batch_size=config.batch_size,
                    initial_weights_sha256=weight_hash,trainer='official OpenPI train_pytorch.train_loop',
                    gpu_budget='actual A100 80 GB; user-authorized removal of 16 GiB cap',
                    final_save_counter_fix=True,optimizer_retention='latest checkpoint only; all policy weights retained')
    try:
        trainer.train_loop(config)
    finally:
        trainer.save_checkpoint=original_save
        logging.getLogger().removeHandler(handler);handler.close()
    checkpoint=config.checkpoint_dir/str(config.num_train_steps)
    if not (checkpoint/'model.safetensors').exists():
        raise RuntimeError('Official GPU trainer did not save the completed final update')
    text=(run.path/'logs/sft.log').read_text()
    import re
    rows=[]
    for match in re.finditer(r'step=(\d+) loss=([\deE+.-]+) lr=([\deE+.-]+) grad_norm=([\deE+.-]+) time=([\deE+.-]+)s',text):
        step,loss,lr,grad,seconds=match.groups()
        rows.append(dict(step=int(step)+1,loss=float(loss),learning_rate=float(lr),
                         grad_norm=float(grad),interval_wall_seconds=float(seconds)))
    if len(rows)!=config.num_train_steps:
        raise RuntimeError('Missing/nonfinite official training diagnostics; inspect sft.log')
    write_json(run.path/'logs/sft_metrics.json',rows)
    run.meta.update(checkpoint=str(checkpoint),completed_updates=len(rows))
    return checkpoint
