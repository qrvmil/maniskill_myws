"""Regression contracts for D0 recovery; fixtures are not scientific evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from maniskill_myws.pld.sac import ResidualSAC, SACConfig
from maniskill_myws.pld.replay_buffer import ReplayBuffer


def batch():
    buf=ReplayBuffer(8,8,7)
    for i in range(8):
        buf.add(np.full(8,i/8),np.zeros(7),np.zeros(7),float(i==7),np.zeros(8),np.zeros(7),i==7)
    return buf.sample(8)


def test_warmup_freezes_actor_alpha_but_trains_critic_and_target():
    torch.set_num_threads(2)
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16))
    actor={k:v.clone() for k,v in agent.actor.state_dict().items()}
    alpha=agent.log_alpha.detach().clone()
    q=[p.clone() for p in agent.q1.parameters()]
    target=[p.clone() for p in agent.q1_target.parameters()]
    for _ in range(4):agent.update(batch(),update_actor=False)
    assert all(torch.equal(v,actor[k]) for k,v in agent.actor.state_dict().items())
    assert torch.equal(alpha,agent.log_alpha)
    assert not agent.actor_opt.state and not agent.alpha_opt.state
    assert any(not torch.equal(a,b) for a,b in zip(q,agent.q1.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(target,agent.q1_target.parameters()))
    agent.update(batch(),update_actor=True)
    agent.update(batch(),update_actor=True)
    assert any(not torch.equal(v,actor[k]) for k,v in agent.actor.state_dict().items())


def test_target_polyak_on_each_critic_update():
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16,tau=.2,actor_update_interval=100))
    before=[p.clone() for p in agent.q1_target.parameters()]
    agent.update(batch())
    for old,q,t in zip(before,agent.q1.parameters(),agent.q1_target.parameters()):
        torch.testing.assert_close(t,.8*old+.2*q)


def test_saturated_tanh_density_remains_accurate_and_finite():
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16,action_scale=.5))
    for p in agent.actor.parameters():p.data.zero_()
    agent.actor.body.net[-1].bias.data[:7]=30
    obs=torch.zeros(2,8);base=torch.zeros(2,7)
    torch.manual_seed(7)
    _,logp=agent.actor.sample(obs,base)
    torch.manual_seed(7)
    raw=torch.distributions.Normal(torch.full((2,7),30.),torch.ones(2,7)).rsample()
    jac=2*(np.log(2)-raw-torch.nn.functional.softplus(-2*raw))+np.log(.5)
    expected=(torch.distributions.Normal(30.,1.).log_prob(raw)-jac).sum(-1,keepdim=True)
    torch.testing.assert_close(logp,expected,atol=1e-4,rtol=1e-5)


def test_strict_serl_loader_rejects_missing_weights(tmp_path):
    from maniskill_myws.pld.serl_encoder import SERLResNet10Encoder
    enc=SERLResNet10Encoder((2,32,32,3),32)
    path=tmp_path/'empty.pt';torch.save({'trunk':{}},path)
    with pytest.raises(ValueError,match='trunk'):
        enc.load_pretrained(path)


def test_serl_trunk_is_frozen_and_heads_trainable():
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16,visual_encoder='serl_resnet10',image_shape=(2,32,32,3)))
    enc=agent.actor.obs_encoder.visual
    assert all(not p.requires_grad for p in enc.trunk.parameters())
    assert any(p.requires_grad for p in enc.heads.parameters())
    x=np.random.default_rng(0).integers(256,size=(2,2,32,32,3),dtype=np.uint8)
    y=enc(agent._images_to_tensor(x))
    assert y.shape==(2,512) # one 256-D latent per camera, as SERL
    y.sum().backward()
    assert all(p.grad is None for p in enc.trunk.parameters())
    assert any(p.grad is not None for p in enc.heads.parameters())


def test_otf_rollout_and_eval_share_seeded_selection_and_rng_isolation():
    from maniskill_myws.pld.libero_runner import residual_policy
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16,action_scale=.5))
    cfg={'otf_rollout_actions':3,'residual_scale':.5}
    policy=residual_policy(agent,cfg,'otf',seed=12)
    obs={'state':np.zeros(8),'images':None};base=np.full(7,.8,np.float32)
    state=torch.get_rng_state().clone()
    unit=policy(obs,base)
    assert torch.equal(state,torch.get_rng_state())
    with torch.random.fork_rng():
        torch.manual_seed(12)
        expected=agent.select_action_otf(obs['state'],base,n_actions=3)
    np.testing.assert_allclose(np.clip(base+.5*unit,-1,1),expected,atol=1e-7)
    assert policy.last_diagnostics['otf_candidates']==4
    assert 'q_base' in policy.last_diagnostics
    with pytest.raises(ValueError):residual_policy(agent,cfg,'unknown',seed=0)


def test_intermediate_checkpoint_only_allowed_for_source_validation():
    from maniskill_myws.pld.libero_protocol import require_specialist_regimen,residual_training_spec
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_otf.json').read_text())
    meta=dict(training_spec=residual_training_spec(cfg),training_steps=42,sac_config={})
    require_specialist_regimen(meta,cfg,source_validation=True)
    with pytest.raises(ValueError):require_specialist_regimen(meta,cfg)
    meta['training_steps']=cfg['online_steps']+100
    with pytest.raises(ValueError):require_specialist_regimen(meta,cfg,source_validation=True)


def test_q_action_set_reuses_features_without_changing_values_or_gradients():
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16,visual_encoder='resnet10',image_shape=(2,32,32,3),visual_latent_dim=16))
    obs=torch.randn(2,8);actions=torch.randn(2,3,7);images=torch.rand(2,6,32,32)
    calls=[]
    hook=agent.q1.obs_encoder.register_forward_hook(lambda m,args,out:calls.append(len(args[0])))
    values=agent._q_for_action_set(agent.q1,obs,actions,images)
    hook.remove()
    assert calls==[2]
    expected=torch.stack([agent.q1(obs,actions[:,i],images)[:,0] for i in range(3)],1)
    torch.testing.assert_close(values,expected,atol=1e-6,rtol=1e-5)
    actual_grad=torch.autograd.grad(values.sum(),agent.q1.q.net[0].weight,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.sum(),agent.q1.q.net[0].weight)[0]
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-5,rtol=1e-5)


def test_source_alignment_batch_and_lora_rank_are_explicit(tmp_path):
    from maniskill_myws.pld.libero_alignment import make_openpi_config
    from openpi.models import gemma
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_otf.json').read_text())
    cfg.update(sft_batch_size=8,sft_save_interval=1000)
    train=make_openpi_config(cfg,repo_id='local/source',workdir=tmp_path,method='lora32',steps=4000)
    assert train.batch_size==8 and train.save_interval==1000
    assert train.data.repo_id=='local/source' and train.data.extra_delta_transform is False
    for variant in ('gemma_2b_lora','gemma_300m_lora'):
        assert all(x.rank==32 for x in gemma.get_config(variant).lora_configs.values())


def test_rollout_records_physical_residual_and_clipping():
    from test_pld_libero import observation
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from maniskill_myws.pld.libero_runner import run_episode
    class Model:
        def reset(self,seed):pass
        def infer(self,obs):return {'actions':np.full((5,7),.9,np.float32)}
    class Env:
        prompt='source'
        def reset(self,seed):return observation(),dict(reset_hash='x',initial_state=np.zeros(2))
        def physics_state(self):return np.zeros(2)
        def step(self,action):return observation(),0.,False,True,dict(success=False)
    row,_=run_episode(Env(),ChunkedBasePolicy(Model()),seed=2000,image_size=4,residual=lambda o,b:np.ones(7))
    assert row['clipped_action_fraction']==1.
    assert row['residual_mean_abs']==.5
    np.testing.assert_allclose(row['executed_residual_mean_abs'],.1,atol=1e-7)
    assert row['residual_per_dim_abs']==[.5]*7


def test_v2_training_budget_requires_full_warmup_and_counts_active_only():
    from maniskill_myws.pld.libero_experiment import training_complete
    cfg={'active_steps':5000,'warmup_episodes':100,'online_steps':27000}
    assert not training_complete(cfg,99,22000,0)
    assert not training_complete(cfg,100,22000,0)
    assert not training_complete(cfg,120,27000,4999)
    assert training_complete(cfg,125,27100,5100)


def test_strict_visual_load_initializes_every_network(tmp_path):
    from maniskill_myws.pld.serl_encoder import SERLTrunk
    trunk=SERLTrunk()
    for p in trunk.parameters():p.data.fill_(.02)
    path=tmp_path/'weights.pt';torch.save({'trunk':trunk.state_dict()},path)
    a=ResidualSAC(SACConfig(8,7,hidden_dim=16,visual_encoder='serl_resnet10',image_shape=(2,32,32,3)))
    counts=a.load_visual_encoder(path)
    assert set(counts)=={'actor','q1','q2','q1_target','q2_target'} and all(v==36 for v in counts.values())
    for visual in a._visual_modules().values():
        for k,v in visual.trunk.state_dict().items():torch.testing.assert_close(v,trunk.state_dict()[k],rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_critic_diagnostics_are_reproducible_and_do_not_change_training_rng(device):
    from maniskill_myws.pld.libero_diagnostics import diagnose_critic
    if device=='cuda' and (not __import__('os').environ.get('PLD_LIBERO_INTEGRATION') or not torch.cuda.is_available()):
        pytest.skip('GPU diagnostic RNG check requires integration runtime')
    a=ResidualSAC(SACConfig(8,7,hidden_dim=16),device=device)
    buf=ReplayBuffer(2,8,7)
    for mc in [.99,1.]:buf.add(np.zeros(8),np.zeros(7),np.zeros(7),1.,np.zeros(8),np.zeros(7),True,mc_return=mc)
    state=torch.get_rng_state().clone();numpy_state=np.random.get_state()
    gpu_state=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=diagnose_critic(a,buf,count=2)
    assert torch.equal(state,torch.get_rng_state())
    if gpu_state is not None:assert torch.equal(gpu_state,torch.cuda.get_rng_state())
    assert np.array_equal(numpy_state[1],np.random.get_state()[1])
    assert result==diagnose_critic(a,buf,count=2)
    assert result['mc_return']['mean']==pytest.approx(.995)
    assert all(k in result for k in ('q_base','q_random_edit','q_mean_edit','q_sampled_edit',
        'random_edit_preference_rate','sampled_edit_preference_rate','entropy','alpha'))
    assert np.isfinite(result['entropy']['mean']) and result['alpha']==1.


@pytest.mark.parametrize('logp,target,direction',[(10.,-3.5,1),(-10.,-3.5,-1)])
def test_temperature_moves_toward_target_entropy(logp,target,direction):
    a=ResidualSAC(SACConfig(8,7,hidden_dim=16,target_entropy=target))
    loss=-(a.log_alpha*(torch.tensor(logp)+a.target_entropy)).mean()
    loss.backward();a.alpha_opt.step()
    assert float(a.log_alpha)*direction>0


def test_calql_critic_only_has_no_actor_gradients():
    a=ResidualSAC(SACConfig(8,7,hidden_dim=16,calql_n_actions=2))
    a.pretrain_critic_calql(batch())
    assert all(p.grad is None for p in a.actor.parameters())


def test_hard_otf_target_masks_terminal_transition():
    a=ResidualSAC(SACConfig(8,7,hidden_dim=16,otf_backup_actions=1,gamma=.99))
    # Constant target twins2,3 => min=2; terminal target must be reward1.
    for q,value in [(a.q1_target,2.),(a.q2_target,3.)]:
        for p in q.parameters():p.data.zero_()
        q.q.net[-1].bias.data.fill_(value)
    b=batch();b.rewards[:]=1.;b.dones[:]=1.
    metrics=a.update(b,update_actor=False)
    assert metrics['target_q']==1.
    b.dones[:]=0.
    # Re-establish constants after preceding Polyak update.
    for q,value in [(a.q1_target,2.),(a.q2_target,3.)]:
        for p in q.parameters():p.data.zero_()
        q.q.net[-1].bias.data.fill_(value)
    metrics=a.update(b,update_actor=False)
    assert metrics['target_q']==pytest.approx(2.98)


def test_training_orchestration_freezes_warmup_and_keeps_real_horizon(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_pld_libero import observation,ChunkModel
    from maniskill_myws.pld import libero_experiment as exp
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_otf.json').read_text())
    cfg.update(device='cpu',rl_image_size=32,batch_size=2,buffer_capacity=16,calql_updates=1,
        calql_n_actions=2,calql_alpha=3.,warmup_episodes=2,active_steps=3,online_steps=7,checkpoint_active_interval=1)
    cfg['source']['horizon']=2
    protocol=Protocol(cfg)
    alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    offline=ReplayBuffer(2,8,7,image_shape=(2,32,32,3))
    im=np.zeros((2,32,32,3),np.uint8)
    for i in range(2):offline.add(np.zeros(8),np.zeros(7),np.zeros(7),1.,np.zeros(8),np.zeros(7),True,1.,images=im,next_images=im)
    path=tmp_path/'offline.npz'
    offline.save(path,kind='aligned_base_success',source=protocol.source,split_hash=protocol.split_hash,
        execution_hash=protocol.execution_hash,alignment_sha256=file_sha256(alignment))
    class Env:
        prompt='source'
        def __init__(self,task,render_size):self.horizon=task['horizon']
        def reset(self,seed):self.i=0;return observation(),dict(reset_hash=str(seed),initial_state=np.zeros(2))
        def physics_state(self):return np.array([self.i,0.])
        def step(self,action):
            assert self.horizon==2
            self.i+=1
            return observation(),float(self.i==2),self.i==2,False,dict(success=self.i==2)
        def close(self):pass
    monkeypatch.setattr(exp,'LiberoEnv',Env)
    for folder in ('logs','checkpoints'): (tmp_path/folder).mkdir()
    run=SimpleNamespace(path=tmp_path,meta={},begin_cuda_phase=lambda:0,
        end_cuda_phase=lambda kind:dict(allocated_bytes=0,reserved_bytes=0))
    args=SimpleNamespace(offline_buffer=path,alignment_manifest=alignment)
    exp.train(cfg,args,run,ChunkedBasePolicy(ChunkModel()),SimpleNamespace(),protocol)
    check=json.loads((tmp_path/'warmup_check.json').read_text())
    assert check['actor_identical'] and check['alpha_identical'] and check['critic_changed']
    rows=json.loads((tmp_path/'training_episodes.json').read_text())
    assert [r['length'] for r in rows]==[2,2,2,2]
    assert [r['warmup'] for r in rows]==[True,True,False,False]
    assert run.meta['active_steps']==4 and run.meta['training_steps']==8
    saved=json.loads(Path(run.meta['checkpoint']).with_suffix('.json').read_text())
    assert saved['sac_config']['calql_alpha']==3.
    assert saved['training_spec']['calql_alpha']==3.


def test_specialist_save_refuses_overwrite(tmp_path):
    from types import SimpleNamespace
    from maniskill_myws.pld.libero_experiment import _save_specialist
    from maniskill_myws.pld.libero_protocol import Protocol
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_otf.json').read_text())
    manifest=tmp_path/'alignment';manifest.write_text('{}')
    agent=ResidualSAC(SACConfig(8,7,hidden_dim=16))
    run=SimpleNamespace(path=tmp_path,meta={})
    _save_specialist(agent,run,Protocol(cfg),manifest,3)
    with pytest.raises(FileExistsError):_save_specialist(agent,run,Protocol(cfg),manifest,3)


@pytest.mark.skipif(not __import__('os').environ.get('PLD_SERL_REFERENCE'),reason='official SERL checkout/weights opt-in')
def test_official_pretrained_serl_features_match_jax():
    import os,pickle,sys
    root=Path(os.environ['PLD_SERL_REFERENCE'])
    sys.path.insert(0,str(root/'serl'/'serl_launcher'))
    from serl_launcher.vision.resnet_v1 import resnetv1_configs
    import jax.numpy as jnp
    from maniskill_myws.pld.serl_encoder import SERLTrunk,convert_serl_params
    from maniskill_myws.pld.libero_protocol import file_sha256
    source=root/'resnet10_params.pkl'
    assert file_sha256(source)=='175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b'
    with source.open('rb') as f:params=pickle.load(f)
    images=np.random.default_rng(11).integers(256,size=(2,128,128,3),dtype=np.uint8)
    reference=np.asarray(resnetv1_configs['resnetv1-10-frozen']().apply({'params':params},jnp.asarray(images)))
    model=SERLTrunk();model.load_state_dict(convert_serl_params(params),strict=True)
    with torch.no_grad():actual=model(torch.from_numpy(images).permute(0,3,1,2).float()/255).permute(0,2,3,1).numpy()
    np.testing.assert_allclose(actual,reference,atol=8e-5,rtol=2e-4)


def test_warmup_review_waits_and_binds_decision_to_diagnostics(tmp_path,monkeypatch):
    from maniskill_myws.pld import libero_experiment as exp
    from maniskill_myws.pld.libero_protocol import file_sha256
    for name in ('warmup_check.json','critic_after_warmup.json'):
        (tmp_path/name).write_text('{}')
    calls=[]
    def approve(_):
        calls.append(True)
        request=json.loads((tmp_path/'warmup_review_request.json').read_text())
        (tmp_path/'warmup_review_decision.json').write_text(json.dumps(dict(
            decision='continue',evidence=request['evidence'],reason='Measured critic reviewed')))
    monkeypatch.setattr(exp.time,'sleep',approve)
    exp.await_warmup_review(tmp_path)
    assert len(calls)==1
    assert json.loads((tmp_path/'warmup_review_request.json').read_text())['evidence']['warmup_check.json']==file_sha256(tmp_path/'warmup_check.json')
    (tmp_path/'critic_after_warmup.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence'):
        exp.await_warmup_review(tmp_path)
    (tmp_path/'warmup_review_decision.json').unlink()
    def stop(_):
        request=json.loads((tmp_path/'warmup_review_request.json').read_text())
        (tmp_path/'warmup_review_decision.json').write_text(json.dumps(dict(
            decision='stop',evidence=request['evidence'],reason='Critic scale invalid')))
    monkeypatch.setattr(exp.time,'sleep',stop)
    with pytest.raises(RuntimeError,match='Critic scale invalid'):
        exp.await_warmup_review(tmp_path)


def test_evaluation_saves_first_failure_pair_after_successful_pairs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import imageio.v2 as imageio
    from maniskill_myws.pld import libero_experiment as exp
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(initial_sim_states=[],initial_observation_sim_states=[])))
    manifest=dict(source_audit=str(audit),source_audit_sha256=file_sha256(audit),aligned_checkpoint='fixture')
    class Env:
        prompt='source';task_id=0
        def __init__(self,*a,**kw):pass
        def close(self):pass
    def episode(env,base,*,seed,**kw):
        row=dict(seed=seed,success=seed==2000,length=1,reset_hash=str(seed),physics_states=[[0.,1.]],residual_mean_abs=0.)
        images=np.zeros((2,4,4,3),np.uint8)
        return row,[dict(action=np.zeros(7),images=images,next_images=images)]
    written=[]
    monkeypatch.setattr(exp,'LiberoEnv',Env);monkeypatch.setattr(exp,'run_episode',episode)
    monkeypatch.setattr(imageio,'mimwrite',lambda path,*a,**kw:written.append(Path(path).name))
    (tmp_path/'eval').mkdir()
    args=SimpleNamespace(mode='zero',validation=False,checkpoint=None,alignment_manifest=str(alignment),episodes=3,videos=1)
    exp.evaluate(cfg,args,SimpleNamespace(path=tmp_path,meta={}),None,SimpleNamespace(),Protocol(cfg),manifest)
    assert len(written)==2
    assert all('_2001_' in name for name in written)
    assert any('_base.mp4' in name for name in written) and any('_residual.mp4' in name for name in written)
