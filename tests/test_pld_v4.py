"""D0 alignment / D1 residual boundaries must fail closed."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from maniskill_myws.pld.libero_protocol import Protocol, file_sha256, directory_manifest, task_key


def config():
    cfg = json.loads(Path('configs/pld_libero/anchor_bowl_v3.json').read_text())
    d0, d1 = cfg['tasks'][:2]
    cfg.pop('source'); cfg.pop('tasks')
    cfg.update(protocol_version=2, base_alignment_task=d0, residual_training_task=d1,
        evaluation_tasks=[d0, d1], base_sanity_seeds=list(range(5000,5050)),
        train_env_seeds=list(range(6000,6100)), validation_env_seeds=list(range(7000,7050)),
        eval_seeds=list(range(8000,8050)), training_scope='D0_BASE_D1_RESIDUAL',
        eval_residual='deterministic_actor', probe_fraction=0.)
    return cfg


def alignment(tmp_path, cfg):
    demo=tmp_path/'d0.h5'; demo.write_bytes(b'd0-only fixture')
    norm=tmp_path/'norm.json'; norm.write_text('{}')
    ckpt=tmp_path/'base'; ckpt.mkdir(); (ckpt/'weights').write_bytes(b'frozen')
    manifest=dict(training_tasks=[task_key(cfg['base_alignment_task'])],
        split_hash=Protocol(cfg).split_hash,
        pretrained_checkpoint='gs://openpi-assets/checkpoints/pi0_base',
        alignment_steps=3001, alignment_data_version='native_post_action_shift_v1',
        demonstrations=[dict(path=str(demo),sha256=file_sha256(demo))],
        normalization=dict(path=str(norm),sha256=file_sha256(norm)),
        aligned_checkpoint=str(ckpt),checkpoint_files=directory_manifest(ckpt))
    path=tmp_path/'alignment.json'; path.write_text(json.dumps(manifest))
    return path, manifest


def test_v4_alignment_accepts_only_d0(tmp_path):
    cfg=config(); protocol=Protocol(cfg); path,manifest=alignment(tmp_path,cfg)
    assert protocol.require_alignment(path)['training_tasks']==[protocol.base_alignment_key]
    manifest['training_tasks']=[task_key(cfg['residual_training_task'])]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='task/split'): protocol.require_alignment(path)


def test_v4_training_accepts_only_d1():
    cfg=config(); protocol=Protocol(cfg)
    protocol.require_training_task(cfg['residual_training_task'])
    with pytest.raises(ValueError): protocol.require_training_task(cfg['base_alignment_task'])
    protocol.require_alignment_task(cfg['base_alignment_task'])
    with pytest.raises(ValueError): protocol.require_alignment_task(cfg['residual_training_task'])


def test_v4_evaluation_allows_exact_registered_tasks():
    cfg=config(); protocol=Protocol(cfg)
    for task in cfg['evaluation_tasks']: protocol.require_evaluation_task(task)
    foreign=json.loads(Path('configs/pld_libero/anchor_bowl_v3.json').read_text())['tasks'][2]
    with pytest.raises(ValueError): protocol.require_evaluation_task(foreign)
    changed=dict(cfg['residual_training_task'],horizon=999)
    with pytest.raises(ValueError): protocol.require_training_task(changed)


def test_v4_split_binds_both_roles_and_all_seed_blocks():
    cfg=config(); original=Protocol(cfg).split_hash
    for role in ('base_alignment_task','residual_training_task'):
        other=copy.deepcopy(cfg)
        other[role]['rationale']+=' changed provenance'
        assert Protocol(other).split_hash!=original
    other=copy.deepcopy(cfg); other['base_sanity_seeds']=[5100]
    assert Protocol(other).split_hash!=original
    other['base_sanity_seeds']=[6000]
    with pytest.raises(ValueError): Protocol(other)


def test_v4_zero_report_uses_d0_sanity_block(tmp_path):
    from maniskill_myws.pld.libero_protocol import require_zero_report
    cfg=config(); protocol=Protocol(cfg); path,_=alignment(tmp_path,cfg)
    report=dict(passed=True,pairs=2,seeds=[5000,5001],
        alignment_sha256=file_sha256(path),split_hash=protocol.split_hash,
        execution_hash=protocol.execution_hash,task=protocol.base_alignment_key)
    zero=tmp_path/'zero.json'; zero.write_text(json.dumps(report))
    require_zero_report(zero,protocol,path)
    report['task']=protocol.residual_training_key; zero.write_text(json.dumps(report))
    with pytest.raises(ValueError): require_zero_report(zero,protocol,path)


def test_v4_replay_requires_d1_provenance(tmp_path):
    from maniskill_myws.pld.libero_experiment import _load_offline
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    cfg=config(); cfg['rl_image_size']=4; protocol=Protocol(cfg)
    manifest,_=alignment(tmp_path,cfg)
    path,replay,metadata=base_replay_fixture(tmp_path,cfg,protocol,manifest)
    assert len(_load_offline(path,cfg,protocol,manifest))==50
    for altered in (dict(metadata,source=protocol.base_alignment_key),
                    {k:v for k,v in metadata.items() if k!='residual_training_task'}):
        replay.save(path,**altered)
        with pytest.raises(ValueError,match='provenance'): _load_offline(path,cfg,protocol,manifest)


def test_v4_evaluation_plan_keeps_retention_hidden_until_selection():
    from types import SimpleNamespace
    from maniskill_myws.pld.libero_experiment import evaluation_plan
    cfg=config(); protocol=Protocol(cfg)
    args=SimpleNamespace(mode='eval',validation=True,distance='D1')
    tasks,seeds,scope=evaluation_plan(cfg,args,protocol)
    assert tasks==[cfg['residual_training_task']] and seeds==cfg['validation_env_seeds']
    assert scope=='residual_validation'
    args.distance='D0'
    with pytest.raises(ValueError): evaluation_plan(cfg,args,protocol)
    args.validation=False
    with pytest.raises(ValueError,match='frozen'): evaluation_plan(cfg,args,protocol)
    tasks,seeds,scope=evaluation_plan(cfg,args,protocol,selected=True)
    assert tasks==[cfg['base_alignment_task']] and seeds==cfg['eval_seeds']
    args.distance='D2'
    with pytest.raises(ValueError): evaluation_plan(cfg,args,protocol,selected=True)
    args.mode='base'; args.distance='D0'
    assert evaluation_plan(cfg,args,protocol)[1]==cfg['base_sanity_seeds']
    args.distance='D1'
    assert evaluation_plan(cfg,args,protocol)[1]==cfg['train_env_seeds']


def test_v4_zero_actions_preserve_rollout_contract():
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from maniskill_myws.pld.libero_runner import run_episode
    from test_pld_libero import observation,ChunkModel
    class Env:
        prompt='D0'; horizon=3
        def reset(self,*,seed):
            self.i=0
            return observation(),dict(reset_hash=str(seed),initial_state=np.zeros(2))
        def physics_state(self): return np.array([self.i])
        def step(self,action):
            self.i+=1
            return observation(),float(self.i==3),self.i==3,False,dict(success=self.i==3)
    base=ChunkedBasePolicy(ChunkModel())
    plain,_=run_episode(Env(),base,seed=5000,image_size=4)
    zero,_=run_episode(Env(),base,seed=5000,image_size=4,residual=lambda o,a:np.zeros(7))
    assert plain['trajectory_hash']==zero['trajectory_hash']
    assert plain['image_hash']==zero['image_hash']


def selection_fixture(tmp_path,success_count=10):
    from maniskill_myws.pld.libero_protocol import paired_summary,residual_training_spec
    cfg=config(); protocol=Protocol(cfg); manifest,_=alignment(tmp_path,cfg)
    checkpoint=tmp_path/'residual.pt'; checkpoint.write_bytes(b'new D1 specialist fixture')
    provenance=dict(source=protocol.residual_training_key,
        base_alignment_task=protocol.base_alignment_key,residual_training_task=protocol.residual_training_key,
        split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,
        alignment_sha256=file_sha256(manifest),checkpoint_sha256=file_sha256(checkpoint),
        training_seed=0,training_steps=27000,active_steps=5000,
        training_spec=residual_training_spec(cfg),sac_config={'action_scale':.5})
    checkpoint.with_suffix('.json').write_text(json.dumps(provenance))
    run=tmp_path/'validation'; (run/'eval').mkdir(parents=True)
    (run/'metadata.json').write_text(json.dumps(dict(status='COMPLETED')))
    (run/'config.json').write_text(json.dumps(dict(cfg,command_options=dict(mode='eval',validation=True,
        distance='D1',eval_policy='deterministic_actor',alignment_manifest=str(manifest),checkpoint=str(checkpoint)))))
    base=[dict(seed=s,reset_hash=str(s),success=False,length=220) for s in cfg['validation_env_seeds']]
    residual=[dict(row,success=i<success_count) for i,row in enumerate(base)]
    task=cfg['residual_training_task']
    (run/'eval'/f"{task['name']}_episodes.json").write_text(json.dumps(dict(base=base,residual=residual)))
    row=dict(source=protocol.residual_training_key,target=protocol.residual_training_key,distance='D1',
        evaluation_scope='residual_validation',evaluation_policy='deterministic_actor',
        alignment_sha256=file_sha256(manifest),checkpoint_sha256=file_sha256(checkpoint),
        residual_checkpoint=str(checkpoint),executed_residual_mean_abs=.1,**paired_summary(base,residual))
    (run/'eval/summary.json').write_text(json.dumps([row]))
    return cfg,run,manifest,checkpoint



def complete_selection_fixture(tmp_path,success_count=10):
    import shutil
    from maniskill_myws.pld.libero_adaptation_selection import d1_stage_record
    cfg,first,manifest,checkpoint=selection_fixture(tmp_path,success_count=success_count)
    runs=[first]
    for milestone in (10000,25000,50000):
        run=tmp_path/f'validation_{milestone}'; shutil.copytree(first,run)
        ckpt=tmp_path/f'residual_{milestone}.pt'; ckpt.write_bytes(str(milestone).encode())
        provenance=json.loads(checkpoint.with_suffix('.json').read_text())
        provenance.update(training_steps=22000+milestone,active_steps=milestone,checkpoint_sha256=file_sha256(ckpt))
        ckpt.with_suffix('.json').write_text(json.dumps(provenance))
        conf=json.loads((run/'config.json').read_text());conf['command_options']['checkpoint']=str(ckpt)
        (run/'config.json').write_text(json.dumps(conf))
        rows=json.loads((run/'eval/summary.json').read_text())
        rows[0].update(residual_checkpoint=str(ckpt),checkpoint_sha256=file_sha256(ckpt))
        (run/'eval/summary.json').write_text(json.dumps(rows));runs.append(run)
    stage=tmp_path/'stage.json';stage.write_text(json.dumps(d1_stage_record(runs,cfg)))
    return cfg,runs,manifest,checkpoint,stage


def base_replay_fixture(tmp_path,cfg,protocol,manifest):
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    replay=ReplayBuffer(50,8,7,image_shape=(2,cfg['rl_image_size'],cfg['rl_image_size'],3))
    im=np.zeros(replay.image_shape,np.uint8)
    for _ in range(50):
        replay.add(np.zeros(8),np.zeros(7),np.zeros(7),1,np.zeros(8),np.zeros(7),True,1,images=im,next_images=im)
    seeds=cfg['train_env_seeds'][:50]
    collection=tmp_path/'collection.json';collection.write_text(json.dumps(dict(attempts=50,successes=50,transitions=50,
        episodes=[dict(seed=seed,success=True,length=1) for seed in seeds])))
    metadata=dict(kind='aligned_base_success',source=protocol.residual_training_key,
        residual_training_task=protocol.residual_training_key,base_alignment_task=protocol.base_alignment_key,
        split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,alignment_sha256=file_sha256(manifest),
        collection_path=str(collection),collection_sha256=file_sha256(collection),successful_seeds=seeds,attempt_seeds=seeds,
        successes=50,attempts=50,gamma=.99)
    path=tmp_path/'replay.npz';replay.save(path,**metadata)
    return path,replay,metadata

def test_v4_selection_uses_only_verified_d1_validation(tmp_path):
    from maniskill_myws.pld.libero_selection import select_source_validation,require_source_selection
    cfg,runs,manifest,checkpoint,stage=complete_selection_fixture(tmp_path)
    run=runs[0]
    record=select_source_validation(runs,cfg,stage_decision=stage,role='residual',policy='deterministic_actor')
    assert record['selection_task']==task_key(cfg['residual_training_task'])
    assert record['frozen'] and record['validation_gain']==.2
    selected=tmp_path/'selection.json'; selected.write_text(json.dumps(record))
    assert require_source_selection(selected,cfg,manifest,checkpoint)==record
    with pytest.raises(ValueError): select_source_validation([run],cfg,role='base')
    with pytest.raises(ValueError): select_source_validation([run],cfg,role='residual',policy='auto')
    record['frozen']=False; selected.write_text(json.dumps(record))
    with pytest.raises(ValueError): require_source_selection(selected,cfg,manifest,checkpoint)
    summary=run/'eval/summary.json'; rows=json.loads(summary.read_text())
    rows[0].update(target=task_key(cfg['base_alignment_task']),distance='D0')
    summary.write_text(json.dumps(rows))
    with pytest.raises(ValueError): select_source_validation([run],cfg,role='residual',policy='deterministic_actor')


def test_v4_selection_rejects_relabelled_or_changed_pair_evidence(tmp_path):
    from maniskill_myws.pld.libero_selection import select_source_validation
    cfg,run,_,_=selection_fixture(tmp_path)
    path=run/'eval'/f"{cfg['residual_training_task']['name']}_episodes.json"
    rows=json.loads(path.read_text()); rows['residual'][0]['reset_hash']='different'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError): select_source_validation([run],cfg,role='residual',policy='deterministic_actor')


def test_progress_audit_reports_reach_grasp_lift_and_plate_progress():
    from maniskill_myws.pld.libero_progress import summarize_progress
    rows=[dict(reach_distance=.2,bowl_plate_distance=.3,bowl_height=.8,grasp=False),
          dict(reach_distance=.01,bowl_plate_distance=.1,bowl_height=.85,grasp=True)]
    result=summarize_progress(rows)
    assert result['minimum_reach_distance']==.01
    assert result['ever_grasped'] and result['ever_lifted_3cm']
    assert result['bowl_plate_progress']==pytest.approx(.2)


def test_v4_online_training_constructs_only_d1_and_saves_role_provenance(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from maniskill_myws.pld import libero_experiment as exp
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    from test_pld_libero import observation,ChunkModel
    torch.set_num_threads(1)
    cfg=config();cfg.update(device='cpu',rl_image_size=4,visual_encoder='none',
        calql_updates=1,calql_n_actions=2,batch_size=2,buffer_capacity=8,
        warmup_episodes=1,active_steps=4,online_steps=6,checkpoint_active_steps=[4],
        stage_review_active_steps=[],warmup_review_required=False)
    cfg['residual_training_task']['horizon']=2
    protocol=Protocol(cfg);manifest,_=alignment(tmp_path,cfg)
    path,_,_=base_replay_fixture(tmp_path,cfg,protocol,manifest)
    constructed=[]
    class Env:
        prompt='D1';horizon=2
        def __init__(self,task,**kw):constructed.append(task)
        def reset(self,*,seed):
            assert seed in cfg['train_env_seeds'];self.i=0
            return observation(),dict(reset_hash=str(seed),initial_state=np.zeros(2))
        def physics_state(self):return np.array([self.i])
        def step(self,action):
            self.i+=1
            return observation(),float(self.i==2),self.i==2,False,dict(success=self.i==2)
        def close(self):pass
    monkeypatch.setattr(exp,'LiberoEnv',Env)
    run_dir=tmp_path/'run';(run_dir/'checkpoints').mkdir(parents=True);(run_dir/'logs').mkdir()
    run=SimpleNamespace(path=run_dir,meta={},begin_cuda_phase=lambda:0,
        end_cuda_phase=lambda kind:dict(allocated_bytes=0,reserved_bytes=0))
    exp.train(cfg,SimpleNamespace(offline_buffer=path,alignment_manifest=manifest,resume_stage=None),
        run,ChunkedBasePolicy(ChunkModel()),SimpleNamespace(),protocol)
    assert constructed==[cfg['residual_training_task']]
    assert run.meta['active_steps']==4
    meta=json.loads(Path(run.meta['checkpoint']).with_suffix('.json').read_text())
    assert meta['base_alignment_task']==protocol.base_alignment_key
    assert meta['residual_training_task']==protocol.residual_training_key


def test_v4_final_summary_accepts_roles_and_preserves_interference(tmp_path):
    from maniskill_myws.pld.libero_protocol import paired_summary
    from maniskill_myws.pld.libero_selection import select_source_validation
    from maniskill_myws.pld.libero_summary import gather_transfer_results
    cfg,vals,manifest,checkpoint,stage=complete_selection_fixture(tmp_path,success_count=0)
    selected=tmp_path/'selected.json'; selected.write_text(json.dumps(
        select_source_validation(vals,cfg,role='residual',policy='deterministic_actor',stage_decision=stage)))
    run=tmp_path/'final'; (run/'eval').mkdir(parents=True)
    args=dict(mode='eval',validation=False,distance='D0',alignment_manifest=str(manifest),
        checkpoint=str(checkpoint),selection_manifest=str(selected))
    (run/'config.json').write_text(json.dumps(dict(cfg,command_options=args)))
    frozen=json.loads(selected.read_text());(run/'frozen_selection.json').write_text(json.dumps(frozen))
    disclosure=dict(selection_manifest_sha256=file_sha256(selected),frozen_selection_sha256=file_sha256(run/'frozen_selection.json'),
        d1_improvement_gate_passed=False,selected_d1_validation_gain=0.,final_result_status='negative_result_gate_failed')
    (run/'metadata.json').write_text(json.dumps(dict(status='COMPLETED',**disclosure)))
    base=[dict(seed=s,reset_hash=str(s),success=i<10,length=220) for i,s in enumerate(cfg['eval_seeds'])]
    residual=[dict(r,success=False) for r in base]
    task=cfg['base_alignment_task']; provenance=json.loads(checkpoint.with_suffix('.json').read_text())
    row=dict(target=task_key(task),checkpoint_sha256=file_sha256(checkpoint),
        alignment_sha256=file_sha256(manifest),evaluation_policy='deterministic_actor',**disclosure,
        **{'residual_'+k:provenance[k] for k in ('training_steps','training_spec','sac_config')},
        **paired_summary(base,residual))
    (run/'eval/summary.json').write_text(json.dumps([row]))
    (run/'eval'/f"{task['name']}_episodes.json").write_text(json.dumps(dict(base=base,residual=residual)))
    result=gather_transfer_results([run])
    assert result['tasks'][0]['delta_SR']==-.2
    assert result['tasks'][0]['base_only_successes']==10
    assert result['tasks'][0]['d1_improvement_gate_passed'] is False
    assert result['tasks'][0]['final_result_status']=='negative_result_gate_failed'
    assert result['tasks'][0]['selection_manifest_sha256']==file_sha256(selected)
    assert result['missing_tasks'][0]['targets']==[task_key(cfg['residual_training_task'])]


def test_zero_discordance_interval_is_not_zero_width():
    from maniskill_myws.pld.libero_summary import gain_interval
    result=gain_interval(dict(episodes=50,residual_only_successes=0,base_only_successes=0,
        paired_bootstrap_95ci=[0.,0.]))
    assert result['gain_95ci'][0]<-.05 and result['gain_95ci'][1]>.05
    assert 'discordance' in result['gain_interval_method']
