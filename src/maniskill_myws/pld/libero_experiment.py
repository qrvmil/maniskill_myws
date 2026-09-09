"""Scientific orchestration. All modes require source-only alignment provenance."""
import json
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .libero_alignment import make_openpi_config
from .libero_artifacts import write_json
from .libero_backend import LiberoEnv, ChunkedBasePolicy, openpi_observation
from .libero_protocol import (Protocol, task_key, file_sha256, paired_summary,
                              require_zero_report, residual_training_spec, require_specialist_regimen)
from .libero_runner import run_episode, insert_trajectory, actor_residual
from .replay_buffer import ReplayBuffer, sample_offline_online
from .sac import ResidualSAC, SACConfig


class AlignedOpenPIModel:
    """One frozen local OpenPI instance with independent seeded sampling noise."""
    def __init__(self, cfg, manifest, run):
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared.normalize import load
        import dataclasses
        from .libero_openpi import fast_loading_config
        train_cfg=make_openpi_config(cfg,repo_id=manifest['repo_id'],workdir=manifest['workdir'],
                                     method=manifest['method'],steps=manifest['alignment_steps'])
        if Path(manifest['aligned_checkpoint'],'model.safetensors').exists():
            train_cfg=dataclasses.replace(train_cfg,model=fast_loading_config(train_cfg.model))
        self.policy=create_trained_policy(train_cfg,manifest['aligned_checkpoint'],
            norm_stats=load(Path(manifest['normalization']['path']).parent),
            pytorch_device=cfg['device'])
        if getattr(self.policy,'_is_pytorch_model',False):
            self.policy._model.requires_grad_(False)
            self.policy._model.eval()
        self.inference_seconds=[]
        self.run=run
        self.noise_shape=(train_cfg.model.action_horizon,train_cfg.model.action_dim)
        self.prompt=''
        self.reset(0)

    def reset(self,seed):
        self.rng=np.random.default_rng(seed)

    def infer(self,raw):
        # Explicit noise avoids advancing global torch/JAX RNG with RL updates.
        noise=self.rng.standard_normal(self.noise_shape).astype(np.float32)
        self.run.begin_cuda_phase()
        started=time.perf_counter()
        result=self.policy.infer(openpi_observation(raw,self.prompt),noise=noise)
        self.inference_seconds.append(time.perf_counter()-started)
        self.run.end_cuda_phase('base_inference')
        return result


def _load_offline(path, cfg, protocol, manifest_path):
    shape=(2,cfg['rl_image_size'],cfg['rl_image_size'],3)
    with np.load(path,allow_pickle=False) as data:
        size=int(data['size'])
    if size < 1:
        raise ValueError('No successful aligned-base transitions')
    buffer=ReplayBuffer(size,8,7,image_shape=shape)
    meta=buffer.load(path)
    if (meta.get('kind')!='aligned_base_success' or meta.get('source')!=protocol.source
            or meta.get('split_hash')!=protocol.split_hash
            or meta.get('execution_hash')!=protocol.execution_hash
            or meta.get('alignment_sha256')!=file_sha256(manifest_path)):
        raise ValueError('Offline replay provenance mismatch; dummy/legacy/foreign replay forbidden')
    if not np.array_equal(buffer.actions[:size],buffer.base_actions[:size]):
        raise ValueError('Offline actions must be actual frozen base actions')
    return buffer


def _save_specialist(agent, run, protocol, manifest_path, steps):
    path=run.path/'checkpoints'/f'residual_step_{steps}.pt'
    agent.save(path)
    provenance={'source':protocol.source,'split_hash':protocol.split_hash,
                'training_seed':protocol.config['training_seed'],
                'alignment_sha256':file_sha256(manifest_path),'execution_hash':protocol.execution_hash,'training_steps':steps,
                'training_spec':residual_training_spec(protocol.config),'sac_config':asdict(agent.config),
                'checkpoint_sha256':file_sha256(path),'checkpoint_selection':'fixed training budget; no unseen metrics'}
    write_json(path.with_suffix('.json'),provenance)
    run.meta['checkpoint']=str(path)
    return path


def _load_specialist(path, cfg, protocol, manifest_path):
    meta=json.loads(Path(path).with_suffix('.json').read_text())
    require_specialist_regimen(meta,cfg)
    if (meta['source']!=protocol.source or meta['split_hash']!=protocol.split_hash
            or meta.get('training_seed')!=cfg['training_seed']
            or meta.get('execution_hash')!=protocol.execution_hash
            or meta['alignment_sha256']!=file_sha256(manifest_path)
            or meta['checkpoint_sha256']!=file_sha256(path)):
        raise ValueError('Specialist checkpoint provenance mismatch')
    agent=ResidualSAC.load(path,device=cfg['device'])
    if json.loads(json.dumps(asdict(agent.config))) != meta['sac_config']:
        raise ValueError('Loaded SAC configuration differs from checkpoint provenance')
    if agent.config.image_shape!=(2,cfg['rl_image_size'],cfg['rl_image_size'],3) or agent.config.visual_encoder!='resnet10':
        raise ValueError('Visual checkpoint observation contract mismatch')
    if agent.config.action_scale!=cfg['residual_scale']:
        raise ValueError('Residual bound differs from checkpoint')
    return agent


def evaluate(cfg,args,run,base,model,protocol,manifest):
    from .libero_protocol import paired_summary
    agent=None if args.mode in ['base','zero'] else _load_specialist(args.checkpoint,cfg,protocol,args.alignment_manifest)
    specialist_meta=json.loads(Path(args.checkpoint).with_suffix('.json').read_text()) if agent else None
    if specialist_meta:
        run.meta['residual_training_steps']=specialist_meta['training_steps']
        run.meta['residual_training_spec']=specialist_meta['training_spec']
        run.meta['residual_sac_config']=specialist_meta['sac_config']
    if args.mode=='eval' and not args.checkpoint:
        raise ValueError('Frozen source specialist required')
    tasks=[cfg['source']] if args.mode in ['base','zero'] else [t for t in cfg['tasks'] if args.distance is None or t['distance']==args.distance]
    if not tasks:
        raise ValueError('No tasks for requested distance')
    audit_path=Path(manifest['source_audit'])
    if file_sha256(audit_path)!=manifest['source_audit_sha256']:
        raise ValueError('Source demonstration audit has changed')
    audit=json.loads(audit_path.read_text())
    demo_states=audit['initial_sim_states']+audit['initial_observation_sim_states']
    zero_checks=[]
    summaries=[]
    for task in tasks:
        env=LiberoEnv(task,render_size=cfg['render_size']);model.prompt=env.prompt
        base_rows=[];residual_rows=[]
        try:
            seeds=cfg['validation_env_seeds'] if args.mode=='zero' or args.validation else cfg['eval_seeds']
            for seed in seeds[:args.episodes]:
                holdout=demo_states if task_key(task)==protocol.source else None
                row,base_tr=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'],demonstration_states=holdout)
                base_rows.append(row)
                if args.mode!='base':
                    residual=(lambda o,a:np.zeros(7)) if args.mode=='zero' else actor_residual(agent)
                    r,res_tr=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'],residual=residual,
                                    residual_scale=cfg['residual_scale'],demonstration_states=holdout)
                    residual_rows.append(r)
                    if args.mode=='zero':
                        errors={k:max(float(np.max(np.abs(np.asarray(x[k],float)-np.asarray(y[k],float)))) for x,y in zip(base_tr,res_tr)) for k in ['action','images','next_images']}
                        same_length=len(base_tr)==len(res_tr)
                        physics_error=float(np.max(np.abs(np.asarray(row['physics_states'])-np.asarray(r['physics_states'])))) if same_length else None
                        passed=(same_length and row['success']==r['success'] and row['reset_hash']==r['reset_hash'] and errors['action']<=1e-5 and physics_error<=1e-5 and errors['images']<=1 and errors['next_images']<=1)
                        zero_checks.append(dict(seed=seed,passed=passed,physics_max_abs=physics_error,**errors))
                        write_json(run.path/'eval/zero_checks.json',zero_checks)
                        if not passed:raise ValueError('Aligned pi0 zero-residual equivalence failed; see zero_checks.json')
                write_json(run.path/'eval'/f"{task['name']}_episodes.json",{'base':base_rows,'residual':residual_rows})
            if residual_rows:
                result=paired_summary(base_rows,residual_rows)
            else:
                result={'episodes':len(base_rows),'seeds':[r['seed'] for r in base_rows],
                        'base_successes':sum(r['success'] for r in base_rows),
                        'SR_base':float(np.mean([r['success'] for r in base_rows])),
                        'base_mean_length':float(np.mean([r['length'] for r in base_rows])),
                        'SR_residual':None,'delta_SR':None}
            summaries.append(dict(source=protocol.source,target=task_key(task),distance=task['distance'],
                                  training_seed=cfg['training_seed'],
                                  checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
                                  alignment_sha256=file_sha256(args.alignment_manifest),
                                  evaluation_scope='source_validation' if args.mode=='zero' or args.validation else 'held_out_evaluation',
                                  task_id=env.task_id,checkpoint=args.checkpoint,
                                  base_checkpoint=manifest['aligned_checkpoint'],residual_checkpoint=args.checkpoint,**result))
            if specialist_meta:
                summaries[-1].update(residual_training_steps=specialist_meta['training_steps'],
                    residual_training_spec=specialist_meta['training_spec'],
                    residual_sac_config=specialist_meta['sac_config'])
            write_json(run.path/'eval/summary.json',summaries)
        finally:
            env.close()
    if args.mode=='zero':
        write_json(run.path/'eval/zero_equivalence.json',dict(passed=all(x['passed'] for x in zero_checks),pairs=len(zero_checks),seeds=[x['seed'] for x in zero_checks],alignment_sha256=file_sha256(args.alignment_manifest),split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,checks=zero_checks))
    if args.mode=='eval':
        buckets={d:float(np.mean([r['delta_SR'] for r in summaries if r['distance']==d]))
                 for d in sorted({r['distance'] for r in summaries})}
        write_json(run.path/'eval/gain_by_distance.json',buckets)
    run.meta.update(episodes=sum(s['episodes'] for s in summaries),
                    rollout_episodes=sum(s['episodes'] for s in summaries)*(1 if args.mode=='base' else 2),
                    checkpoint=args.checkpoint or manifest['aligned_checkpoint'])


def collect(cfg,args,run,base,model,protocol):
    protocol.require_training_task(cfg['source'])
    shape=(2,cfg['rl_image_size'],cfg['rl_image_size'],3)
    buffer=ReplayBuffer(args.successes*cfg['source']['horizon'],8,7,image_shape=shape)
    env=LiberoEnv(cfg['source'],render_size=cfg['render_size']);model.prompt=env.prompt
    rows=[];successes=0
    try:
        for seed in cfg['train_env_seeds'][:args.max_attempts]:
            row,transitions=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'])
            rows.append(row)
            if row['success']:
                insert_trajectory(buffer,transitions,gamma=.99)
                successes+=1
            write_json(run.path/'eval/collection.json',{'attempts':len(rows),'successes':successes,
                'transitions':len(buffer),'base_SR':successes/len(rows),'episodes':rows})
            if successes>=args.successes:
                break
        if not successes:
            raise RuntimeError('No successful aligned-base rollouts; Cal-QL gate remains closed')
        buffer.save(run.path/'offline.npz',kind='aligned_base_success',source=protocol.source,
                    split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,alignment_sha256=file_sha256(args.alignment_manifest),gamma=.99,
                    attempts=len(rows),successes=successes)
        run.meta.update(episodes=len(rows),successes=successes,transitions=len(buffer),
                        achieved_requested_successes=successes>=args.successes)
    finally:
        env.close()


def train(cfg,args,run,base,model,protocol):
    import torch
    protocol.require_training_task(cfg['source'])
    offline=_load_offline(args.offline_buffer,cfg,protocol,args.alignment_manifest)
    shape=(2,cfg['rl_image_size'],cfg['rl_image_size'],3)
    online=ReplayBuffer(cfg['buffer_capacity'],8,7,image_shape=shape)
    agent=ResidualSAC(SACConfig(8,7,action_scale=cfg['residual_scale'],visual_encoder='resnet10',
        image_shape=shape,calql_n_actions=cfg['calql_n_actions'],otf_backup_actions=cfg['otf_backup_actions'],
        target_entropy=cfg.get('target_entropy')),device=cfg['device'])
    log=run.path/'logs/updates.jsonl'
    def update(kind,batch,step):
        allocated_before=run.begin_cuda_phase()
        t=time.perf_counter()
        metrics=agent.pretrain_critic_calql(batch) if kind=='calql' else agent.update(batch)
        if torch.cuda.is_available():torch.cuda.synchronize()
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f'Nonfinite {kind} losses: {metrics}')
        memory=run.end_cuda_phase(kind)
        with log.open('a') as f:
            f.write(json.dumps(dict(kind=kind,step=step,seconds=time.perf_counter()-t,
                cuda_peak_allocated_bytes=memory['allocated_bytes'],
                cuda_peak_reserved_bytes=memory['reserved_bytes'],
                cuda_additional_peak_allocated_bytes=memory['allocated_bytes']-allocated_before,**metrics))+'\n')
    for i in range(cfg['calql_updates']):
        update('calql',offline.sample(cfg['batch_size']),i)
    env=LiberoEnv(cfg['source'],render_size=cfg['render_size']);model.prompt=env.prompt
    env_steps=0;episode=0;rows=[]
    def residual(obs,base_action):
        if cfg['otf_rollout_actions']:
            a=agent.select_action_otf(obs['state'],base_action,images=obs['images'],n_actions=cfg['otf_rollout_actions'])
            return np.clip((a-base_action)/cfg['residual_scale'],-1,1)
        return agent.select_delta(obs['state'],base_action,images=obs['images'],deterministic=False)/cfg['residual_scale']
    def on_transition(tr):
        nonlocal env_steps
        online.add(**tr)
        env_steps+=1
        update('sac',sample_offline_online(offline,online,cfg['batch_size'],offline_fraction=.5),env_steps)
    try:
        while env_steps<cfg['online_steps']:
            # The final incomplete rollout is a finite-horizon cutoff, logged.
            env.horizon=min(cfg['source']['horizon'],cfg['online_steps']-env_steps)
            seed=cfg['train_env_seeds'][episode%len(cfg['train_env_seeds'])]
            warmup=episode<cfg.get('warmup_episodes',5)
            row,_=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'],
                residual=None if warmup else residual,residual_scale=cfg['residual_scale'],on_transition=on_transition)
            rows.append(dict(env_steps=env_steps,warmup=warmup,**row));episode+=1
            write_json(run.path/'training_episodes.json',rows)
            if episode%10==0:
                _save_specialist(agent,run,protocol,args.alignment_manifest,env_steps)
        _save_specialist(agent,run,protocol,args.alignment_manifest,env_steps)
        run.meta.update(episodes=len(rows),training_steps=env_steps,calql_updates=cfg['calql_updates'])
    finally:
        env.close()


def scientific(cfg,args,run):
    import torch
    # Fail missing simulator dependencies before expensive model construction.
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    validation_started=time.perf_counter()
    protocol=Protocol(cfg);manifest=protocol.require_alignment(args.alignment_manifest)
    run.meta['alignment_validation_seconds']=time.perf_counter()-validation_started
    if args.mode not in ['base','zero']:
        require_zero_report(args.zero_report,protocol,args.alignment_manifest)
    torch.manual_seed(cfg['training_seed']);np.random.seed(cfg['training_seed']);torch.set_num_threads(2)
    if args.mode=='train' and not args.offline_buffer:
        raise ValueError('Successful aligned-base --offline-buffer required')
    if args.mode=='eval' and not args.checkpoint:
        raise ValueError('Frozen source --checkpoint required')
    load_started=time.perf_counter()
    model=AlignedOpenPIModel(cfg,manifest,run)
    run.meta['base_load_seconds']=time.perf_counter()-load_started
    base=ChunkedBasePolicy(model,replan_steps=cfg['replan_steps'])
    try:
        if args.mode in ['base','zero','eval']:
            evaluate(cfg,args,run,base,model,protocol,manifest)
        elif args.mode=='collect':
            collect(cfg,args,run,base,model,protocol)
        elif args.mode=='train':
            train(cfg,args,run,base,model,protocol)
    finally:
        write_json(run.path/'logs/base_inference_seconds.json',model.inference_seconds)
