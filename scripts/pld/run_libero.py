#!/usr/bin/env python3
"""LIBERO PLD entrypoint. Smoke policies are explicitly non-VLA diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import numpy as np

from maniskill_myws.pld.libero_artifacts import RunArtifacts, write_json, memory_fraction
from maniskill_myws.pld.libero_backend import LiberoEnv, ChunkedBasePolicy
from maniskill_myws.pld.libero_protocol import Protocol
from maniskill_myws.pld.libero_runner import run_episode, insert_trajectory


class SmokeModel:
    """Bounded deterministic diagnostic, never a substitute for aligned pi0."""
    def reset(self, seed):
        self.rng=np.random.default_rng(seed)
    def infer(self, obs):
        chunk=self.rng.uniform(-.02,.02,(5,7)).astype(np.float32)
        chunk[:,6]=-1
        return {'actions':chunk}


def smoke(cfg, args, run):
    import torch
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    torch.set_num_threads(2)
    torch.manual_seed(cfg['training_seed'])
    np.random.seed(cfg['training_seed'])
    task=dict(cfg['source'])
    if args.max_steps:
        task['horizon']=args.max_steps
    env=LiberoEnv(task,render_size=cfg['render_size'])
    base=ChunkedBasePolicy(SmokeModel(),replan_steps=cfg['replan_steps'])
    rows=[]
    try:
        for seed in cfg['eval_seeds'][:args.episodes]:
            a,tr=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'])
            b,zero_tr=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'],
                            residual=lambda o,a:np.zeros(7),residual_scale=cfg['residual_scale'])
            differences={k:max(float(np.max(np.abs(np.asarray(x[k],float)-np.asarray(y[k],float)))) for x,y in zip(tr,zero_tr)) for k in ['obs','next_obs','action','images','next_images']}
            write_json(run.path/f'eval/zero_difference_{seed}.json',differences)
            write_json(run.path/f'eval/zero_pair_{seed}.json',{'base':a,'zero':b})
            if (a['trajectory_hash'] != b['trajectory_hash'] or a['reset_hash'] != b['reset_hash']
                    or differences['images'] > 1 or differences['next_images'] > 1):
                raise AssertionError(f'Zero residual changes trajectory: {differences}')
            c,_=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'],
                            residual=lambda o,a:np.full(7,.2),residual_scale=cfg['residual_scale'])
            rows.append({'base_dummy':a,'zero_residual_dummy':b,'bounded_residual_dummy':c})
        write_json(run.path/'eval/episodes.json',rows)
        shape=(2,cfg['rl_image_size'],cfg['rl_image_size'],3)
        buffer=ReplayBuffer(len(tr),8,7,image_shape=shape)
        insert_trajectory(buffer,tr)
        buffer.save(run.path/'diagnostic_replay.npz',kind='dummy_non_scientific',source=task['name'])
        sac=SACConfig(8,7,hidden_dim=256,visual_encoder='resnet10',image_shape=shape,
                       action_scale=cfg['residual_scale'],calql_n_actions=2,otf_backup_actions=0)
        agent=ResidualSAC(sac,device=cfg['device'])
        metrics=[]
        for kind in ['calql','calql','sac','sac']:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start=time.perf_counter()
            m=agent.pretrain_critic_calql(buffer.sample(2)) if kind=='calql' else agent.update(buffer.sample(2))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            assert all(np.isfinite(v) for v in m.values()),m
            metrics.append(dict(kind=kind,seconds=time.perf_counter()-start,**m))
        write_json(run.path/'losses.json',metrics)
        agent.save(run.path/'checkpoints/diagnostic_residual.pt')
        run.meta.update(episodes=len(rows)*3,zero_equivalence_pairs=len(rows),
                        diagnostic_only=True,checkpoint=str(run.path/'checkpoints/diagnostic_residual.pt'))
        print(json.dumps({'zero_equivalence_pairs':len(rows),'finite_updates':len(metrics),
                          'diagnostic_only':True,'output':str(run.path)}))
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['smoke','base','zero','collect','train','eval'])
    p.add_argument('--config',default='configs/pld_libero/anchor_bowl.json')
    p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=2)
    p.add_argument('--validation',action='store_true',help='Source base only: use source-validation seeds for alignment selection')
    p.add_argument('--max-steps',type=int,default=None,help='Only truncate non-scientific smoke')
    p.add_argument('--alignment-manifest')
    p.add_argument('--zero-report')
    p.add_argument('--checkpoint')
    p.add_argument('--offline-buffer')
    p.add_argument('--distance',choices=['D0','D1','D2','D3','D4','D5'])
    p.add_argument('--successes',type=int,default=50)
    p.add_argument('--max-attempts',type=int,default=100)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    protocol=Protocol(cfg)
    if args.validation and args.mode!='base':
        p.error('--validation is only supported for source base evaluation')
    if args.mode != 'smoke':
        protocol.require_alignment(args.alignment_manifest)
        if args.max_steps is not None:
            p.error('Only smoke may override horizon')
    allowed_seeds=cfg['validation_env_seeds'] if args.mode=='zero' or args.validation else cfg['eval_seeds']
    if args.episodes<1 or args.episodes>len(allowed_seeds):
        p.error('episodes must be within registered evaluation seed set')
    cfg['command_options']=vars(args)
    with RunArtifacts(args.output,cfg) as run:
        import torch
        if cfg['device'].startswith('cuda'):
            total=torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(memory_fraction(cfg['gpu_budget_gib'],total))
        if args.mode=='smoke':
            smoke(cfg,args,run)
        else:
            from maniskill_myws.pld.libero_experiment import scientific
            scientific(cfg,args,run)

if __name__=='__main__':
    main()
