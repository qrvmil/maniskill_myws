"""Contract tests: keep simulator/model dependencies outside fast unit tests."""
import json
from pathlib import Path
import numpy as np
import pytest


def observation():
    im = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    return {'agentview_image': im, 'robot0_eye_in_hand_image': im + 100,
            'robot0_eef_pos': np.array([1., 2., 3.]),
            'robot0_eef_quat': np.array([0., 0., 0., 1.]),
            'robot0_gripper_qpos': np.array([.02, -.02])}


def test_camera_rotation_order_and_proprioception():
    from maniskill_myws.pld.libero_backend import convert_observation
    out = convert_observation(observation(), 'pick bowl', image_size=4)
    np.testing.assert_array_equal(out['images'][0, 0, 0], [45, 46, 47])
    np.testing.assert_array_equal(out['images'][1, 0, 0], [145, 146, 147])
    np.testing.assert_allclose(out['state'], [1, 2, 3, 0, 0, 0, .02, -.02])
    assert out['images'].shape == (2, 4, 4, 3)
    assert out['images'].dtype == np.uint8
    assert out['prompt'] == 'pick bowl'


def test_quaternion_conversion_does_not_mutate_observation():
    from maniskill_myws.pld.libero_backend import quat_to_axisangle
    q = np.array([0, 0, np.sqrt(.5), np.sqrt(.5)])
    original = q.copy()
    np.testing.assert_allclose(quat_to_axisangle(q), [0, 0, np.pi/2], atol=1e-6)
    np.testing.assert_array_equal(q, original)


def test_bounded_actions_zero_equivalence_and_invalid_inputs():
    from maniskill_myws.pld.libero_backend import ActionContract
    c = ActionContract(.5)
    a = np.array([-.9, -.3, 0, .2, .9, 1, -1], np.float32)
    np.testing.assert_array_equal(c.compose(a, np.zeros(7)), a)
    b = c.compose(a, np.ones(7))
    np.testing.assert_allclose(b, [-.4, .2, .5, .7, 1, 1, -.5])
    assert np.max(np.abs(b-a)) <= .5 + 1e-6
    np.testing.assert_array_equal(c.to_env(c.from_env(a)), a)
    for bad in [np.zeros(8), np.full(7, np.nan), np.full(7, 1.1)]:
        with pytest.raises(ValueError):
            c.to_env(bad)
    for scale in [-.1, 0, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            ActionContract(scale)
    with pytest.raises(ValueError):
        c.compose(a, np.full(7, 2))


class ChunkModel:
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
    def infer(self, obs):
        return {'actions': np.repeat(self.rng.uniform(-1, 1, (5, 1)), 7, axis=1)}


def test_chunk_advancement_replan_and_seed_reset():
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    p = ChunkedBasePolicy(ChunkModel(), replan_steps=2)
    p.reset(7)
    a, b, c = [p.act({}) for _ in range(3)]
    expected = np.random.default_rng(7).uniform(-1, 1, 10)
    np.testing.assert_allclose([a[0], b[0], c[0]], expected[[0, 1, 5]])
    p.reset(7)
    np.testing.assert_array_equal(p.act({}), a)
    with pytest.raises(ValueError):
        ChunkedBasePolicy(ChunkModel(), replan_steps=0)


def test_protocol_rejects_leakage_and_seed_overlap():
    from maniskill_myws.pld.libero_protocol import Protocol
    cfg = json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text())
    p = Protocol(cfg)
    p.require_training_task(cfg['source'])
    with pytest.raises(ValueError):
        p.require_training_task(cfg['tasks'][1])
    with pytest.raises(ValueError):
        p.require_alignment(None)
    cfg['eval_seeds'] = [1000]
    with pytest.raises(ValueError):
        Protocol(cfg)


def test_paired_summary_preserves_negative_transfer():
    from maniskill_myws.pld.libero_protocol import paired_summary
    a = [{'seed': 3, 'reset_hash':'x', 'success':True, 'length':2},
         {'seed': 4, 'reset_hash':'y', 'success':True, 'length':2}]
    b = [{'seed': 3, 'reset_hash':'x', 'success':False, 'length':4},
         {'seed': 4, 'reset_hash':'y', 'success':True, 'length':2}]
    s = paired_summary(a, b)
    assert s['SR_base'] == 1 and s['SR_residual'] == .5 and s['delta_SR'] == -.5
    b[0]['seed'] = 5
    with pytest.raises(ValueError):
        paired_summary(a, b)


def test_visual_replay_calql_sac_and_deterministic_checkpoint(tmp_path):
    import torch
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    buf = ReplayBuffer(4, 8, 7, image_shape=(2, 32, 32, 3))
    im = np.random.randint(256, size=(2,32,32,3), dtype=np.uint8)
    for i in range(4):
        buf.add(np.zeros(8), np.zeros(7), np.zeros(7), float(i==3),
                np.zeros(8), np.zeros(7), i==3, .99**(3-i), images=im, next_images=im)
    batch = buf.sample(2)
    assert batch.images.dtype == np.uint8 and batch.images.shape == (2,2,32,32,3)
    agent = ResidualSAC(SACConfig(8, 7, hidden_dim=32, visual_encoder='resnet10',
        image_shape=(2,32,32,3), visual_latent_dim=32, calql_n_actions=2, action_scale=.5))
    before = {k:v.clone() for k,v in agent.actor.state_dict().items()}
    metrics = agent.pretrain_critic_calql(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(before[k], v) for k,v in agent.actor.state_dict().items())
    metrics = agent.update(batch)
    assert 'actor_loss' in metrics and all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(before[k], v) for k,v in agent.actor.state_dict().items())
    a = agent.select_action(np.zeros(8), np.zeros(7), images=im)
    assert np.max(np.abs(a)) <= .5
    agent.save(tmp_path/'agent.pt')
    loaded = ResidualSAC.load(tmp_path/'agent.pt')
    np.testing.assert_array_equal(a, loaded.select_action(np.zeros(8), np.zeros(7), images=im))


def test_main_otf_temperature_gradient_preserves_unit_residual_target():
    """Catch applying a unit-coordinate entropy target to scaled actor density."""
    import torch
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    cfg = json.loads(Path('configs/pld_libero/anchor_bowl_otf.json').read_text())
    torch.manual_seed(0)
    scaled = ResidualSAC(SACConfig(8, 7, hidden_dim=32,
        action_scale=cfg['residual_scale'], target_entropy=cfg['target_entropy']))
    unit = ResidualSAC(SACConfig(8, 7, hidden_dim=32, action_scale=1, target_entropy=-3.5))
    unit.actor.load_state_dict(scaled.actor.state_dict())
    unit.actor.scale.fill_(1)
    obs, base = torch.zeros(16, 8), torch.zeros(16, 7)
    torch.manual_seed(4)
    delta, scaled_logp = scaled.actor.sample(obs, base)
    torch.manual_seed(4)
    residual, unit_logp = unit.actor.sample(obs, base)
    torch.testing.assert_close(delta, cfg['residual_scale'] * residual)
    # The inherited +1e-6 Jacobian stabilizer introduces a small approximation.
    for agent, logp in [(scaled, scaled_logp), (unit, unit_logp)]:
        loss = -(agent.log_alpha * (logp + agent.target_entropy).detach()).mean()
        loss.backward()
    torch.testing.assert_close(scaled.log_alpha.grad, unit.log_alpha.grad, atol=2e-3, rtol=0)


def test_specialist_cannot_be_evaluated_under_a_different_training_regimen(tmp_path):
    from types import SimpleNamespace
    from maniskill_myws.pld.libero_experiment import _save_specialist, _load_specialist
    from maniskill_myws.pld.libero_protocol import Protocol
    from maniskill_myws.pld.sac import ResidualSAC, SACConfig
    cfg = json.loads(Path('configs/pld_libero/anchor_bowl_rl_smoke.json').read_text())
    cfg.update(device='cpu', rl_image_size=32)
    (tmp_path/'checkpoints').mkdir()
    alignment = tmp_path/'alignment.json'; alignment.write_text('source fixture')
    run = SimpleNamespace(path=tmp_path, meta={})
    agent = ResidualSAC(SACConfig(8, 7, hidden_dim=32, action_scale=.5,
        visual_encoder='resnet10', image_shape=(2,32,32,3), visual_latent_dim=32))
    checkpoint = _save_specialist(agent, run, Protocol(cfg), alignment, 8)
    loaded = _load_specialist(checkpoint, cfg, Protocol(cfg), alignment)
    assert loaded.config.action_scale == agent.config.action_scale
    for changes in [dict(online_steps=50000), dict(otf_rollout_actions=1),
                    dict(warmup_episodes=100), dict(calql_updates=1000)]:
        other = dict(cfg, **changes)
        with pytest.raises(ValueError, match='training regimen'):
            _load_specialist(checkpoint, other, Protocol(other), alignment)
    partial = _save_specialist(agent, run, Protocol(cfg), alignment, 4)
    with pytest.raises(ValueError, match='training budget'):
        _load_specialist(partial, cfg, Protocol(cfg), alignment)


def test_rollout_advances_cached_action_once_and_records_next_action():
    from maniskill_myws.pld.libero_runner import run_episode
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    class Env:
        prompt = 'pick bowl'
        def reset(self, *, seed):
            self.i = 0
            return observation(), {'reset_hash':'fixed', 'initial_state':np.zeros(2)}
        def physics_state(self):
            return np.array([self.i,0.,0.])
        def step(self, a):
            self.i += 1
            return observation(), float(self.i==3), self.i==3, False, {'success':self.i==3}
    p = ChunkedBasePolicy(ChunkModel(), replan_steps=2)
    row, traj = run_episode(Env(), p, seed=7, image_size=4)
    assert row['success'] and row['length'] == 3
    np.testing.assert_array_equal(traj[0]['next_base_action'], traj[1]['base_action'])
    np.testing.assert_array_equal(traj[1]['next_base_action'], traj[2]['base_action'])
    np.testing.assert_array_equal(traj[2]['next_base_action'], np.zeros(7))
    np.testing.assert_array_equal(traj[0]['action'], traj[0]['base_action'])


@pytest.mark.skipif(__import__('os').environ.get('PLD_LIBERO_INTEGRATION') != '1', reason='real MuJoCo opt-in')
def test_real_libero_reset_step_zero_equivalence():
    from maniskill_myws.pld.libero_runner import run_episode
    from maniskill_myws.pld.libero_backend import LiberoEnv, ChunkedBasePolicy
    cfg = json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text())
    task = dict(cfg['source'], horizon=8)
    env = LiberoEnv(task, render_size=64)
    try:
        p = ChunkedBasePolicy(ChunkModel(), replan_steps=2)
        a, at = run_episode(env,p,seed=3000,image_size=32)
        b, bt = run_episode(env,p,seed=3000,image_size=32,residual=lambda o,base:np.zeros(7),residual_scale=.5)
        assert a['reset_hash'] == b['reset_hash']
        for i,(x,y) in enumerate(zip(at,bt)):
            for k in ['obs','next_obs','action','base_action']:
                np.testing.assert_array_equal(x[k],y[k],err_msg=f'step={i} key={k}')
            for k in ['images','next_images']:
                # EGL can differ by one uint8 intensity level on reset.
                assert np.max(np.abs(x[k].astype(int)-y[k].astype(int))) <= 1
        np.testing.assert_allclose(a['physics_states'],b['physics_states'],rtol=0,atol=1e-5)
        assert a['trajectory_hash'] == b['trajectory_hash']
        with pytest.raises(RuntimeError):
            env.step(np.zeros(7))
    finally:
        env.close()


def test_source_demo_audit_rejects_foreign_task(tmp_path):
    import h5py
    from maniskill_myws.pld.libero_alignment import audit_source_h5
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text())
    path=tmp_path/'demo.hdf5'
    with h5py.File(path,'w') as f:
        data=f.create_group('data')
        data.attrs['bddl_file_name']='libero_goal/open_the_middle_drawer_of_the_cabinet.bddl'
    with pytest.raises(ValueError,match='source'):
        audit_source_h5(path,cfg)
    with h5py.File(path,'a') as f:
        f['data'].attrs['bddl_file_name']=cfg['source']['suite']+'/'+cfg['source']['name']+'.bddl'
        g=f['data'].create_group('demo_0')
        g.create_dataset('actions',data=np.zeros((3,7)))
        g.create_dataset('states',data=np.zeros((3,10)))
    summary=audit_source_h5(path,cfg)
    assert summary['num_demonstrations']==1 and summary['transitions']==3


def test_scientific_commands_fail_closed_without_alignment(tmp_path):
    import os,subprocess,sys
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text())
    cfg['device']='cpu'
    config_path=tmp_path/'cpu_config.json';config_path.write_text(json.dumps(cfg))
    for mode in ['base','collect','train','eval']:
        p=subprocess.run([sys.executable,'scripts/pld/run_libero.py',mode,
                          '--config',str(config_path),'--output',str(tmp_path/mode)],
                         env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
        assert p.returncode != 0
        assert 'aligned-base manifest' in p.stderr


def test_dataset_binding_rejects_other_repo_and_changed_files(tmp_path):
    from maniskill_myws.pld.libero_alignment import verify_dataset_binding
    from maniskill_myws.pld.libero_protocol import directory_manifest
    root=tmp_path/'local'/'seen';root.mkdir(parents=True)
    (root/'data.txt').write_text('source data')
    audit={'repo_id':'local/seen','root':str(root),'dataset_files':directory_manifest(root)}
    verify_dataset_binding(audit,repo_id='local/seen',root=root)
    with pytest.raises(ValueError):
        verify_dataset_binding(audit,repo_id='physical-intelligence/libero',root=root)
    (root/'data.txt').write_text('other data')
    with pytest.raises(ValueError):
        verify_dataset_binding(audit,repo_id='local/seen',root=root)


def test_directory_manifest_detects_replaced_checkpoint(tmp_path):
    from maniskill_myws.pld.libero_protocol import directory_manifest,verify_directory
    (tmp_path/'weights').write_bytes(b'base1')
    m=directory_manifest(tmp_path)
    verify_directory(tmp_path,m)
    (tmp_path/'weights').write_bytes(b'base2')
    with pytest.raises(ValueError):verify_directory(tmp_path,m)


def test_cpu_optimizer_preserves_all_parameters_and_matches_adamw():
    import torch
    from maniskill_myws.pld.libero_cpu_sft import cpu_optimizer_step
    torch.manual_seed(4)
    a=torch.nn.Linear(3,2);b=torch.nn.Linear(3,2);b.load_state_dict(a.state_dict())
    opt_a=torch.optim.AdamW(a.parameters(),lr=.01,foreach=False)
    opt_b=torch.optim.AdamW(b.parameters(),lr=.01,foreach=False)
    for _ in range(2):
        x=torch.ones(2,3)
        a(x).square().mean().backward();b(x).square().mean().backward()
        cpu_optimizer_step(a,opt_a,grad_clip_norm=0)
        opt_b.step();opt_b.zero_grad(set_to_none=True)
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert all(v.device.type=='cpu' for state in opt_a.state.values() for v in state.values() if torch.is_tensor(v))


def test_zero_report_bound_to_base_and_initial_state_holdout(tmp_path):
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256,assert_held_out,require_zero_report
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text());p=Protocol(cfg)
    manifest=tmp_path/'alignment.json';manifest.write_text('original')
    report=tmp_path/'zero.json'
    report.write_text(json.dumps({'passed':True,'alignment_sha256':file_sha256(manifest),
        'split_hash':p.split_hash,'execution_hash':p.execution_hash,'seeds':[2000,2001],'pairs':2}))
    require_zero_report(report,p,manifest)
    changed=dict(cfg,replan_steps=cfg['replan_steps']+1)
    with pytest.raises(ValueError):require_zero_report(report,Protocol(changed),manifest)
    manifest.write_text('replaced')
    with pytest.raises(ValueError):require_zero_report(report,p,manifest)
    with pytest.raises(ValueError):assert_held_out([2,1,2,3],[[7,1,2,3]])
    assert_held_out([2,1,2,4],[[7,1,2,3]])


@pytest.mark.skipif(__import__('importlib').util.find_spec('openpi') is None,reason='run in OpenPI venv')
def test_official_openpi_action_normalization_roundtrip():
    from maniskill_myws.pld.libero_backend import ActionContract
    from openpi.transforms import Normalize,Unnormalize
    from openpi.shared.normalize import NormStats
    # Model z-scores must be inverted exactly once before bounded OSC actions.
    stats={'actions':NormStats(mean=np.linspace(-.2,.2,7),std=np.linspace(.1,.7,7))}
    actions=np.linspace(-1,1,35).reshape(5,7).astype(np.float32)
    normalized=Normalize(stats)({'actions':actions.copy()})
    assert np.max(np.abs(normalized['actions']))>1
    recovered=Unnormalize(stats)(normalized)['actions']
    np.testing.assert_allclose(recovered,actions,rtol=0,atol=1e-5)
    contract=ActionContract(.5)
    for row in recovered:np.testing.assert_allclose(contract.to_env(np.clip(row,-1,1)),row,atol=1e-5)


def test_resident_cpu_optimizer_matches_move_model_adamw():
    import copy,torch
    from maniskill_myws.pld.libero_cpu_sft import ResidentCPUOptimizer,cpu_optimizer_step
    torch.manual_seed(8)
    reference=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,2))
    resident=copy.deepcopy(reference)
    opt=torch.optim.AdamW(reference.parameters(),lr=.001,foreach=False)
    mirror=ResidentCPUOptimizer(resident,lr=.001,foreach=False)
    for _ in range(3):
        x=torch.randn(3,4);target=torch.randn(3,2)
        for model in [reference,resident]:((model(x)-target)**2).mean().backward()
        cpu_optimizer_step(reference,opt)
        mirror.step(resident)
        for a,b in zip(reference.parameters(),resident.parameters(),strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    restored=ResidentCPUOptimizer(copy.deepcopy(resident),lr=.001,foreach=False)
    restored.optimizer.load_state_dict(mirror.optimizer.state_dict())
    assert len(restored.optimizer.state)==len(mirror.optimizer.state)


def test_transfer_summary_preserves_negative_gain_and_rejects_duplicates(tmp_path):
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256,paired_summary,task_key,residual_training_spec
    from maniskill_myws.pld.libero_summary import gather_transfer_results
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl.json').read_text());p=Protocol(cfg)
    run=tmp_path/'run';(run/'eval').mkdir(parents=True)
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'unit test fixture, not a real checkpoint')
    alignment=tmp_path/'alignment.json';alignment.write_text('unit test source alignment fixture')
    cp_hash=file_sha256(checkpoint);base_hash=file_sha256(alignment)
    checkpoint.with_suffix('.json').write_text(json.dumps(dict(source=p.source,training_seed=0,
        checkpoint_sha256=cp_hash,alignment_sha256=base_hash,split_hash=p.split_hash,execution_hash=p.execution_hash,
        training_spec=residual_training_spec(cfg),training_steps=50000,sac_config={'action_scale':.5})))
    cfg['command_options']=dict(mode='eval',checkpoint=str(checkpoint),alignment_manifest=str(alignment))
    (run/'config.json').write_text(json.dumps(cfg));(run/'metadata.json').write_text(json.dumps(dict(status='COMPLETED')))
    base=[dict(seed=3000,reset_hash='a',success=True,length=10),dict(seed=3001,reset_hash='b',success=False,length=10)]
    residual=[dict(x,success=False) for x in base]
    row=dict(target=task_key(cfg['source']),checkpoint_sha256=cp_hash,alignment_sha256=base_hash,
        residual_training_steps=50000,residual_training_spec=residual_training_spec(cfg),
        residual_sac_config={'action_scale':.5},**paired_summary(base,residual))
    (run/'eval/summary.json').write_text(json.dumps([row]))
    (run/'eval'/f"{cfg['source']['name']}_episodes.json").write_text(json.dumps(dict(base=base,residual=residual)))
    summary=gather_transfer_results([run])
    assert summary['tasks'][0]['delta_SR']==-.5
    assert summary['buckets'][0]['mean_gain']==-.5
    assert len(summary['missing_tasks'][0]['targets'])==8
    with pytest.raises(ValueError,match='Duplicate'):gather_transfer_results([run,run])
    other=dict(cfg,online_steps=8)
    (run/'config.json').write_text(json.dumps(other))
    with pytest.raises(ValueError,match='training regimen'):gather_transfer_results([run])
    (run/'config.json').write_text(json.dumps(cfg))
    (run/'metadata.json').write_text(json.dumps(dict(status='FAILED')))
    with pytest.raises(ValueError,match='completed'):gather_transfer_results([run])


def test_native_post_action_observations_use_next_action_label():
    from maniskill_myws.pld.libero_alignment import native_observation_action_indices
    # states after actions 0,1,2 must predict the following action 1,2,3.
    actions=np.array([10,20,30,40])
    pairs=[(i,actions[j]) for i,j in native_observation_action_indices(len(actions))]
    assert pairs==[(0,20),(1,30),(2,40)]
    with pytest.raises(ValueError):list(native_observation_action_indices(1))


def test_official_gpu_sft_final_checkpoint_counts_completed_updates():
    from dataclasses import dataclass
    from maniskill_myws.pld.libero_gpu_sft import save_completed_update
    @dataclass
    class Config:
        num_train_steps:int=2
        save_interval:int=500
    calls=[]
    def native_save(model,optimizer,step,config,is_main,data):
        if (step%config.save_interval==0 and step>0) or step==config.num_train_steps-1:
            calls.append(step)
    for step in (1,2):
        save_completed_update(native_save,None,None,step,Config(),True,None)
    assert calls==[2]


def test_gpu_checkpoint_retention_preserves_weights_and_latest_optimizer(tmp_path):
    from maniskill_myws.pld.libero_gpu_sft import retain_latest_optimizer
    for step in (500,1000):
        p=tmp_path/str(step);p.mkdir()
        (p/'model.safetensors').write_bytes(b'weights')
        (p/'optimizer.pt').write_bytes(b'optimizer')
    removed=retain_latest_optimizer(tmp_path,1000)
    assert removed==['500/optimizer.pt']
    assert (tmp_path/'500/model.safetensors').read_bytes()==b'weights'
    assert (tmp_path/'1000/optimizer.pt').exists()
    with pytest.raises(ValueError):retain_latest_optimizer(tmp_path,1500)


def test_gpu_budget_fraction_is_float_at_full_device_capacity():
    from maniskill_myws.pld.libero_artifacts import memory_fraction
    total=80*1024**3
    for budget in (16,80,100):
        fraction=memory_fraction(budget,total)
        assert isinstance(fraction,float)
        assert 0<fraction<=1
    assert memory_fraction(16,total)==.2
    assert memory_fraction(80,total)==1.
    with pytest.raises(ValueError):memory_fraction(0,total)


def test_phase_memory_keeps_overall_peak_after_resets():
    from types import SimpleNamespace
    from maniskill_myws.pld.libero_artifacts import RunArtifacts
    class CUDA:
        allocated=20
        reserved=24
        def is_available(self):return True
        def max_memory_allocated(self):return self.allocated
        def max_memory_reserved(self):return self.reserved
        def synchronize(self):pass
        def reset_peak_memory_stats(self):self.allocated=5;self.reserved=8
        def memory_allocated(self):return 5
    run=RunArtifacts.__new__(RunArtifacts)
    run.torch=SimpleNamespace(cuda=CUDA());run.meta={};run.cuda_peaks={}
    assert run.begin_cuda_phase()==5
    run.torch.cuda.allocated=7
    assert run.end_cuda_phase('sac')['allocated_bytes']==7
    assert run.cuda_peaks=={'allocated_bytes':20,'reserved_bytes':24}
    assert run.meta['cuda_phase_peaks']['sac']['allocated_bytes']==7
