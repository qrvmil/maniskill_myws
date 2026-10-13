"""Train one independent nested-task policy; save full raw and EMA weights."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import numpy as np
import torch
from data import ROOT, WORK, TRAIN_SETS
from prepare import sha256, write_json
from training import TaskData, masked_l1, update_ema, training_lock

RECIPE = dict(updates=3001,batch_size=8,seed=0,lr=5e-5,warmup_steps=300,
              betas=[.9,.95],eps=1e-8,weight_decay=1e-10,max_grad_norm=1.,ema_decay=.999)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(value, path):
    tmp = Path(str(path)+'.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def build_initial_model():
    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import TurboVLA
    from turbovla.training.trainer import load_interaction_from_init_checkpoint, load_text_projection_from_init_checkpoint
    config = TurboVLAConfig()
    config.text.model_name_or_path = str(WORK/'assets'/'bert')
    config.text.padding_length = 21
    layout = json.loads((ROOT/'TurboVLA/experiments/libero/configs/online_text_layout.json').read_text())
    config.text.padding_length_by_instruction = layout['padding_length_by_instruction']
    config.vision.model_name_or_path = str(WORK/'assets'/'dinov3')
    torch.manual_seed(RECIPE['seed'])
    model = TurboVLA(config)
    ckpt = str(WORK/'assets'/'groundingdino_swint_ogc.pth')
    reports = dict(interaction=load_interaction_from_init_checkpoint(model,ckpt),
                   text_projection=load_text_projection_from_init_checkpoint(model,ckpt))
    for section,prefix in [('interaction','vision_language_interaction.'),('text_projection','text_encoder.text_projection.')]:
        report = reports[section]
        expected = sum(name.startswith(prefix) for name in model.state_dict())
        if report['mapped'] != expected or report['skipped_shape'] or report['skipped_missing']:
            raise ValueError(f'Incomplete original GroundingDINO initialization: {section}: {report}, expected={expected}')
    return model,reports


def train(variant, smoke_steps=None):
    run=WORK/variant/('smoke' if smoke_steps is not None else 'train')
    with training_lock(run):
        return _train(variant,smoke_steps)


def _train(variant, smoke_steps=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the experiment')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    random.seed(RECIPE['seed'])
    np.random.seed(RECIPE['seed'])
    torch.manual_seed(RECIPE['seed'])
    torch.cuda.manual_seed_all(RECIPE['seed'])
    asset_manifest = json.loads((WORK/'assets'/'manifest.json').read_text())
    data_manifest = json.loads((WORK/'data_manifest.json').read_text())
    tasks = TRAIN_SETS[variant]
    stats_path = WORK/variant/'stats.json'
    stats = json.loads(stats_path.read_text())['train']
    if stats['training_tasks'] != tasks or stats['source_hashes'] != {t:data_manifest['hashes'][t] for t in tasks}:
        raise ValueError('Training/normalization provenance mismatch')
    for t in tasks:
        if sha256(WORK/'data'/f'{t}.hdf5') != data_manifest['hashes'][t]:
            raise ValueError(f'Data changed: {t}')
    for asset in ('dinov3','bert'):
        for name,expected in asset_manifest[asset]['files'].items():
            if sha256(WORK/'assets'/asset/name) != expected:
                raise ValueError(f'Original asset changed: {asset}/{name}')
    if sha256(WORK/'assets'/'groundingdino_swint_ogc.pth') != asset_manifest['groundingdino']['sha256']:
        raise ValueError('GroundingDINO asset changed')
    model,reports = build_initial_model()
    initial_hash = tensor_hash(model.state_dict())
    initial_path = WORK/'initialization.json'
    if initial_path.exists() and json.loads(initial_path.read_text())['state_sha256'] != initial_hash:
        raise ValueError('A/B/C initial weights differ')
    write_json(initial_path,dict(state_sha256=initial_hash,seed=RECIPE['seed'],assets=asset_manifest,reports=reports))
    run = WORK/variant/('smoke' if smoke_steps is not None else 'train')
    run.mkdir(parents=True,exist_ok=True)
    config = dict(variant=variant,tasks=tasks,recipe=RECIPE,initial_state_sha256=initial_hash,
                  stats_sha256=sha256(stats_path),data_manifest_sha256=sha256(WORK/'data_manifest.json'),
                  turbo_revision='a3365e8f1787d30bafd99d2806edbf4dc82b5370',
                  policy_config=model.config.to_dict(),smoke_steps=smoke_steps)
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
        raise ValueError('Existing training recipe changed')
    write_json(run/'config.json',config)
    if (run/'complete.json').exists():
        raise FileExistsError(f'Training already complete: {run}; verify artifacts instead of retraining')
    model.cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=RECIPE['lr'],betas=tuple(RECIPE['betas']),
                                  eps=RECIPE['eps'],weight_decay=RECIPE['weight_decay'])
    ema = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    step_start = 0
    resume_path = run/'resume.pt'
    if resume_path.exists():
        saved = torch.load(resume_path,map_location='cuda')
        if saved['config'] != config:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        ema = saved['ema']
        step_start = saved['step']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
        del saved
    dataset = TaskData(tasks,stats,WORK)
    max_steps = smoke_steps if smoke_steps is not None else RECIPE['updates']
    try:
        with (run/'train.jsonl').open('a') as log:
            for step in range(step_start,max_steps):
                start = time.perf_counter()
                pixels,instructions,states,actions,masks = dataset.batch(step,RECIPE['batch_size'],RECIPE['seed'])
                pixels,states,actions,masks = [x.cuda() for x in (pixels,states,actions,masks)]
                lr = RECIPE['lr']*min(1.,(step+1)/RECIPE['warmup_steps'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                prediction = model(instructions,{'dinov3':pixels},states)
                loss = masked_l1(prediction,actions,masks)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step+1}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params,RECIPE['max_grad_norm'],error_if_nonfinite=True)
                optimizer.step()
                update_ema(ema,model,RECIPE['ema_decay'])
                row = dict(step=step+1,loss=loss.item(),grad_norm=float(norm),lr=lr,
                           wall_seconds=time.perf_counter()-start,max_vram_mib=torch.cuda.max_memory_allocated()/2**20)
                log.write(json.dumps(row)+'\n')
                log.flush()
                if step == 0 or (step+1)%20 == 0:
                    print(json.dumps(row),flush=True)
                if (step+1)%500 == 0 or step+1 == max_steps:
                    atomic_save(dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),ema=ema,
                        step=step+1,config=config,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),resume_path)
        raw = {n:v.detach().cpu() for n,v in model.state_dict().items()}
        ema_state = {n:(ema[n].detach().cpu() if n in ema else v) for n,v in raw.items()}
        checkpoint = dict(model_state_dict=raw,ema_model_state_dict=ema_state,
                          model_config=model.config.to_dict(),step=max_steps,experiment=config)
        final = run/'final.pth'
        atomic_save(checkpoint,final)
        write_json(run/'complete.json',dict(step=max_steps,variant=variant,checkpoint_sha256=sha256(final),
            stats_sha256=sha256(stats_path),initial_state_sha256=initial_hash,
            primary_weights='ema_model_state_dict',training_tasks=tasks,smoke_only=smoke_steps is not None))
        print('TRAINING_COMPLETE',variant,max_steps,flush=True)
    finally:
        dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=list(TRAIN_SETS),required=True)
    parser.add_argument('--smoke-steps',type=int)
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error('--smoke-steps must be positive')
    train(args.variant,args.smoke_steps)
