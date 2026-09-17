#!/usr/bin/env python3
"""Run the preregistered base-only audit without touching historical RL gates."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_sanity import (UPDATES, evaluation_seeds, completed_updates,
    require_d0_binding, PromptCheckedPolicy, wilson, model_contract, save_selected_update, summarize_rows)
from maniskill_myws.pld.libero_protocol import file_sha256, task_key, directory_manifest
from maniskill_myws.pld.libero_artifacts import write_json

AUDIT=json.loads((ROOT/'configs/pld_libero/pi0_sanity_audit.json').read_text())
V4=json.loads((ROOT/AUDIT['d0_config']).read_text())
WORK=Path('/workspace/audit-run')
DATA=Path('/workspace/audit-data/lerobot/local/pi0_audit_d0')
REPO_ID='local/pi0_audit_d0'


def train_config():
    from maniskill_myws.pld.libero_alignment import make_openpi_config
    cfg=make_openpi_config(V4,repo_id=REPO_ID,workdir=WORK/'sft',method='lora32',steps=3001)
    # Save triggers only; no optimizer/model/dataloader change. All selected saves retained.
    return dataclasses.replace(cfg,save_interval=1,keep_period=1,
        weight_loader=dataclasses.replace(cfg.weight_loader,params_path='/workspace/audit-checkpoints/pi0_base/params'))


def bind():
    from maniskill_myws.pld.libero_alignment import verify_dataset_binding
    audit=json.loads((DATA/'source_audit.json').read_text())
    verify_dataset_binding(audit,repo_id=REPO_ID,root=DATA)
    norm=train_config().assets_dirs/REPO_ID/'norm_stats.json'
    provenance=json.loads(norm.with_name('normalization_provenance.json').read_text())
    if provenance['source_audit_sha256']!=file_sha256(DATA/'source_audit.json'):
        raise ValueError('Normalization source differs')
    if provenance['statistics_sha256']!=file_sha256(norm):
        raise ValueError('Normalization hash differs')
    b=dict(training_tasks=[audit['task']],num_demonstrations=audit['num_demonstrations'],
           transitions=audit['transitions'],normalization_path=str(norm),
           normalization_sha256=file_sha256(norm),repo_id=REPO_ID,
           pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_base',
           source_audit_sha256=file_sha256(DATA/'source_audit.json'),
           source_h5_sha256=audit['sha256'],source_h5=audit['path'])
    require_d0_binding(b,AUDIT)
    write_json(WORK/'d0_binding.json',b)
    return b


def train():
    import jax
    import logging
    logging.basicConfig(level=logging.INFO)
    b=bind()
    cfg=train_config()
    (WORK/'training_config.txt').write_text(repr(cfg))
    spec=importlib.util.spec_from_file_location('audit_openpi_train',ROOT/'third_party/openpi/scripts/train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    state={}
    orig_dir=trainer._checkpoints.initialize_checkpoint_dir
    orig_loader=trainer._data_loader.create_data_loader
    orig_init=trainer.init_train_state
    orig_save=trainer._checkpoints.save_state
    def capture_dir(*a,**kw):
        result=orig_dir(*a,**kw);state['manager']=result[0];return result
    def capture_loader(*a,**kw):
        result=orig_loader(*a,**kw);state['loader']=result;return result
    def save(manager,train_state,loader,loop_index):
        updates=save_selected_update(orig_save,manager,train_state,loader,loop_index)
        if updates is None:return
        manager.wait_until_finished()
        write_json(WORK/'sft'/f'update_{updates}.json',dict(optimizer_updates=updates,
            checkpoint_directory=str(cfg.checkpoint_dir/str(updates)),upstream_loop_index=loop_index,
            normalization_sha256=b['normalization_sha256']))
        print(f'AUDIT_SAVED optimizer_updates={updates}',flush=True)
    def capture_initial(*a,**kw):
        result=orig_init(*a,**kw)
        initial=result[0];jax.block_until_ready(initial)
        if int(initial.step)!=0:raise ValueError('Initial state has optimizer updates')
        orig_save(state['manager'],initial,state['loader'],0)
        state['manager'].wait_until_finished()
        write_json(WORK/'sft/update_0.json',dict(optimizer_updates=0,
            checkpoint_directory=str(cfg.checkpoint_dir/'0'),upstream_loop_index=None,
            normalization_sha256=b['normalization_sha256']))
        print('AUDIT_SAVED optimizer_updates=0',flush=True)
        return result
    trainer._checkpoints.initialize_checkpoint_dir=capture_dir
    trainer._data_loader.create_data_loader=capture_loader
    trainer.init_train_state=capture_initial
    trainer._checkpoints.save_state=save
    trainer.main(cfg)
    for updates in UPDATES:
        checkpoint=cfg.checkpoint_dir/str(updates)
        norm=checkpoint/'assets'/REPO_ID/'norm_stats.json'
        if file_sha256(norm)!=b['normalization_sha256']:raise ValueError('Checkpoint normalization drift')
    write_json(WORK/'training_complete.json',dict(optimizer_updates=3001,checkpoints=list(UPDATES)))


def evaluate(model_name, run, *, video_replay=False):
    import numpy as np
    import torch
    from maniskill_myws.pld.libero_runtime import configure_base_inference, require_base_inference_runtime
    configure_base_inference(V4);require_base_inference_runtime(V4)
    from maniskill_myws.pld.libero_backend import LiberoEnv,ChunkedBasePolicy
    from maniskill_myws.pld.libero_runner import run_episode
    from maniskill_myws.pld.libero_experiment import AlignedOpenPIModel
    from maniskill_myws.pld.libero_sanity_videos import select_video_rows,save_rollout_video,validate_replayed_row
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared.normalize import load
    from openpi.training.config import get_config
    from types import SimpleNamespace
    official=model_name=='positive_control'
    model_id=model_name if model_name in ('positive_control','lora_initialization') else int(model_name)
    seeds=evaluation_seeds(AUDIT,model_id)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    out=WORK/'eval'/str(model_id)
    out.mkdir(parents=True,exist_ok=True)
    if video_replay:
        if not (out/'complete.json').exists():raise ValueError('Video replay requires completed original evaluation')
    elif (out/'complete.json').exists():raise FileExistsError(out/'complete.json')
    record_out=out/'video_replay' if video_replay else out
    record_out.mkdir(parents=True,exist_ok=True)
    if (record_out/'complete.json').exists():raise FileExistsError(record_out/'complete.json')
    if official:
        cfg=get_config('pi05_libero')
        ckpt=Path('/workspace/audit-checkpoints/pi05_libero')
        norm=ckpt/'assets/physical-intelligence/libero/norm_stats.json'
        kwargs={}  # MUST load official statistics from checkpoint assets.
    else:
        b=json.loads((WORK/'d0_binding.json').read_text());require_d0_binding(b,AUDIT)
        cfg=train_config()
        contract=model_contract(AUDIT,model_id,b)
        if contract['weight_source']=='official_pretrained':
            from openpi.models.pi0_config import Pi0Config
            cfg=dataclasses.replace(cfg,model=Pi0Config())
            ckpt=Path('/workspace/audit-checkpoints/pi0_base')
            norm=Path(b['normalization_path'])
        else:
            update=contract['checkpoint_updates']
            ckpt=cfg.checkpoint_dir/str(update)
            saved=json.loads((WORK/'sft'/f'update_{update}.json').read_text())
            if saved['optimizer_updates']!=update:raise ValueError('Wrong checkpoint counter')
            norm=ckpt/'assets'/REPO_ID/'norm_stats.json'
        if file_sha256(norm)!=b['normalization_sha256']:raise ValueError('Wrong checkpoint normalization')
        kwargs=dict(norm_stats=load(norm.parent))
    model=AlignedOpenPIModel.__new__(AlignedOpenPIModel)
    model.prompt='';model.inference_seconds=[]
    model.run=run
    model.noise_shape=(cfg.model.action_horizon,cfg.model.action_dim)
    write_json(record_out/'checkpoint_params_sha256.json',directory_manifest(ckpt/'params'))
    started=time.time()
    policy=create_trained_policy(cfg,ckpt,**kwargs)
    expected_task_prompt=['']
    checked=PromptCheckedPolicy(policy,lambda:expected_task_prompt[0])
    model.policy=checked
    base=ChunkedBasePolicy(model,replan_steps=V4['replan_steps'])
    provenance=dict(model=str(model_id),checkpoint=str(ckpt),config=repr(cfg),
        normalization_path=str(norm),normalization_sha256=file_sha256(norm),
        normalization_scheme='q01_q99' if official else 'mean_std',
        action_horizon=cfg.model.action_horizon,action_dim=cfg.model.action_dim,
        audit_script_sha256=file_sha256(__file__),
        checkpoint_params_manifest_sha256=file_sha256(record_out/'checkpoint_params_sha256.json'),
        model_load_seconds=time.time()-started,seed_block=list(seeds))
    write_json(record_out/'binding.json',provenance)
    video_manifest=[]
    for task in AUDIT['tasks']:
        env=LiberoEnv(task,render_size=V4['render_size'])
        model.prompt=env.prompt
        expected=task['name'].replace('_',' ')
        expected_task_prompt[0]=expected
        if env.prompt!=expected:raise ValueError('Simulator language differs from registered prompt')
        references={}
        task_seeds=seeds
        if video_replay:
            references={r['seed']:r for r in select_video_rows(json.loads((out/(task['distance']+'_episodes.json')).read_text()))}
            task_seeds=tuple(references)
        rows=[];raw_chunks=[];first_actions=[];resets=[]
        original_infer=model.infer
        captured=[]
        def capture(raw):
            result=original_infer(raw)
            if not captured:captured.append(np.array(result['actions'],copy=True))
            return result
        model.infer=capture
        try:
            for seed in task_seeds:
                captured.clear()
                row,transitions=run_episode(env,base,seed=seed,image_size=V4['rl_image_size'],record_progress=True,
                    demonstration_states=(json.loads((DATA/'source_audit.json').read_text())['initial_sim_states'] +
                        json.loads((DATA/'source_audit.json').read_text())['initial_observation_sim_states'])
                        if task['distance']=='D0' and not official else None)
                row['prompt']=env.prompt
                row['ever_reached_10cm']=row['task_progress']['minimum_reach_distance']<AUDIT['reach_threshold_m']
                row.pop('physics_states')  # hashes and initial states retained; no large redundant arrays in Git
                rows.append(row)
                write_json(record_out/(task['distance']+'_episodes.json'),rows)
                if video_replay:validate_replayed_row(references[seed],row)
                if seed in {r['seed'] for r in select_video_rows(rows)}:
                    video_manifest.append(save_rollout_video(out/'videos',str(model_id),task['distance'],row,transitions,
                        origin='deterministic_replay' if video_replay else 'evaluation'))
                    write_json(out/'videos/manifest.json',video_manifest)
                if not video_replay and task['distance']=='D1' and seed in seeds[:10]:
                    raw_chunks.append(captured[0]);first_actions.append(np.stack([t['action'] for t in transitions[:5]]));resets.append(row['reset_hash'])
                print(f'AUDIT_{"VIDEO_REPLAY" if video_replay else "EVAL"} model={model_id} task={task["distance"]} seed={seed} success={row["success"]} length={row["length"]}',flush=True)
            if raw_chunks:
                np.savez(out/'d1_first_actions.npz',chunks=np.stack(raw_chunks),executed=np.stack(first_actions),seeds=seeds[:10],reset_hashes=resets)
            if not video_replay:write_json(out/(task['distance']+'_summary.json'),summarize_rows(rows))
        finally:
            model.infer=original_infer;env.close()
    write_json(record_out/'prompt_evidence.json',checked.evidence)
    write_json(record_out/'complete.json',dict(model=str(model_id),wall_seconds=time.time()-started,
        inference_calls=checked.calls,normalization_sha256=file_sha256(norm)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['bind','train','eval'])
    parser.add_argument('--model',default=None)
    parser.add_argument('--video-replay',action='store_true',help='Replay selected completed episodes for videos; never replace metrics')
    args=parser.parse_args()
    if args.mode=='eval':args.model=args.model or 'positive_control'
    elif args.model is not None or args.video_replay:
        parser.error('--model and --video-replay apply only to eval')
    os.environ['HF_LEROBOT_HOME']=str(DATA.parents[1])
    os.environ.setdefault('MUJOCO_GL','egl')
    if args.mode=='eval':
        from maniskill_myws.pld.libero_runtime import configure_base_inference
        configure_base_inference(V4)
    from maniskill_myws.pld.libero_artifacts import RunArtifacts
    run_name=args.mode + ('_'+args.model if args.mode=='eval' else '') + ('_video_replay' if args.video_replay else '')
    with RunArtifacts(WORK/'runtime'/run_name,vars(args)) as run:
        snapshot=run.path/'audit_source'
        for source in [*ROOT.glob('scripts/pld/*pi0*.py'),
                       ROOT/'src/maniskill_myws/pld/libero_sanity.py',
                       ROOT/'src/maniskill_myws/pld/libero_sanity_videos.py',
                       ROOT/'configs/pld_libero/pi0_sanity_audit.json',
                       ROOT/AUDIT['d0_config']]:
            target=snapshot/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        write_json(run.path/'audit_source_sha256.json',directory_manifest(snapshot))
        if args.mode=='train':train()
        elif args.mode=='bind':bind()
        else:evaluate(args.model,run,video_replay=args.video_replay)

if __name__=='__main__':main()
