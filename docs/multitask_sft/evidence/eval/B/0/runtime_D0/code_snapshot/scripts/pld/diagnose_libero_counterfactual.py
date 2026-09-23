#!/usr/bin/env python3
"""Controlled D0 training-state action ranking; never consumes target resets."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from maniskill_myws.pld.libero_runtime import configure_base_inference
from maniskill_myws.pld.libero_protocol import Protocol,require_zero_report,file_sha256
from maniskill_myws.pld.libero_artifacts import RunArtifacts,write_json
from maniskill_myws.pld.libero_counterfactual import branch_rollout,ranking_summary


def main():
    p=argparse.ArgumentParser()
    for name in ('config','alignment-manifest','zero-report','checkpoint','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());configure_base_inference(cfg)
    import torch
    from maniskill_myws.pld.libero_experiment import AlignedOpenPIModel,_load_specialist
    from maniskill_myws.pld.libero_backend import LiberoEnv,ChunkedBasePolicy
    from maniskill_myws.pld.libero_runner import run_episode
    torch.set_num_threads(2)
    protocol=Protocol(cfg);manifest=protocol.require_alignment(args.alignment_manifest)
    require_zero_report(args.zero_report,protocol,args.alignment_manifest)
    with RunArtifacts(args.output,dict(cfg,command_options=vars(args))) as run:
        agent=_load_specialist(args.checkpoint,cfg,protocol,args.alignment_manifest,source_validation=True)
        model=AlignedOpenPIModel(cfg,manifest,run);base=ChunkedBasePolicy(model,replan_steps=cfg['replan_steps'])
        env=LiberoEnv(cfg['source'],render_size=cfg['render_size']);model.prompt=env.prompt
        rows=[];strata={True:0,False:0};rng=np.random.default_rng(9472)
        try:
            for seed in cfg['train_env_seeds'][:20]:
                row,tr=run_episode(env,base,seed=seed,image_size=cfg['rl_image_size'])
                success=row['success']
                if strata[success]>=2:continue
                strata[success]+=1
                for fraction in (.3,.6):
                    index=min(len(tr)-1,int(len(tr)*fraction));prefix=[x['action'] for x in tr[:index]]
                    state=tr[index]['obs'];im=tr[index]['images'];b=tr[index]['base_action']
                    with torch.no_grad(),torch.random.fork_rng(devices=[agent.device.index or 0] if agent.device.type=='cuda' else []):
                        torch.manual_seed(seed*1000+index)
                        x=torch.as_tensor(state,device=agent.device)[None];bt=torch.as_tensor(b,device=agent.device)[None]
                        it=agent._images_to_tensor(im[None])
                        det,_=agent.actor.sample(x,bt,it,deterministic=True)
                        samples=[agent.actor.sample(x,bt,it)[0][0].cpu().numpy() for _ in range(2)]
                        small=rng.normal(size=7);small=.05*small/max(np.max(np.abs(small)),1e-6)
                        edits=[np.zeros(7),det[0].cpu().numpy(),small,-small,*samples]
                        actions=np.stack([np.clip(b+d,-1,1).astype('f') for d in edits])
                        at=torch.as_tensor(actions,device=agent.device)[None]
                        q=torch.minimum(agent._q_for_action_set(agent.q1,x,at,it),agent._q_for_action_set(agent.q2,x,at,it))[0].cpu().tolist()
                    branches=[]
                    for action in actions:
                        branches.append(branch_rollout(env,base,seed=seed,prefix=prefix,
                            candidate=lambda o,base,a=action:a,image_size=cfg['rl_image_size']))
                    repeat=branch_rollout(env,base,seed=seed,prefix=prefix,candidate=lambda o,b:b,image_size=cfg['rl_image_size'])
                    if repeat['trajectory_hash']!=branches[0]['trajectory_hash']:
                        raise RuntimeError('Counterfactual reconstruction changes unedited base continuation')
                    for branch in branches:
                        if branch['branch_state']!=repeat['branch_state'] or branch['prefix_hash']!=repeat['prefix_hash']:
                            raise RuntimeError('Counterfactual candidates do not share exact prefix/state')
                        np.testing.assert_array_equal(branch['base_action'],b)
                    rows.append(dict(seed=seed,trajectory_success=success,index=index,
                        candidate_names=['base','deterministic','small_positive','small_negative','sample1','sample2'],
                        q=q,returns=[z['empirical_return'] for z in branches],branches=branches,
                        repeated_base_identical=True))
                    write_json(run.path/'counterfactual.json',dict(rows=rows,summary=ranking_summary(rows),
                        checkpoint_sha256=file_sha256(args.checkpoint),state_selection='first2 success and first2 failed base trajectories in train seeds1000–1019; .3/.6 trajectory length'))
                if min(strata.values())>=2:break
            import matplotlib;matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(8,3))
            for row in rows:
                axes[0].scatter(row['q'],row['returns'],s=12,alpha=.5)
                axes[1].scatter(np.asarray(row['q'])[1:]-row['q'][0],np.asarray(row['returns'])[1:]-row['returns'][0],s=12,alpha=.5)
            axes[0].set(xlabel='Critic Q',ylabel='One edit, then base return')
            axes[1].set(xlabel='Q(edit) − Q(base)',ylabel='Empirical return difference')
            axes[1].axhline(0,color='gray',lw=.5);axes[1].axvline(0,color='gray',lw=.5)
            fig.tight_layout();fig.savefig(run.path/'calibration.png',dpi=150);plt.close(fig)
            run.meta.update(states=len(rows),success_trajectories=strata[True],failed_trajectories=strata[False],checkpoint=args.checkpoint)
        finally:env.close()

if __name__=='__main__':main()
