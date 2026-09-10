"""Behavior contracts added for V3; legacy defaults remain V2-loadable."""
import numpy as np
import pytest
import torch
from maniskill_myws.pld.sac import ResidualSAC, SACConfig
from maniskill_myws.pld.replay_buffer import ReplayBatch


def agent(**kw):
    return ResidualSAC(SACConfig(8, 7, hidden_dim=16, action_scale=.5,
        residual_actor_impl='serl_uniform_std', residual_density='unit',
        actor_q_reduction='mean', **kw))


def batch():
    rng=np.random.default_rng(0)
    return ReplayBatch(obs=rng.normal(size=(8,8)).astype('f'), actions=np.zeros((8,7),'f'),
        base_actions=np.zeros((8,7),'f'), rewards=np.ones(8,'f'),
        next_obs=rng.normal(size=(8,8)).astype('f'), next_base_actions=np.zeros((8,7),'f'),
        dones=np.ones(8,'f'), mc_returns=np.ones(8,'f'))


def test_uniform_std_is_state_independent_and_learnable():
    a=agent(); obs=torch.randn(4,8); base=torch.randn(4,7)
    mean, std=a.actor(obs,base)
    assert mean.shape==(4,7)
    assert torch.equal(std,torch.zeros_like(std))
    std.sum().backward()
    assert torch.equal(a.actor.log_stds.grad,torch.full((7,),4.))
    with torch.no_grad():a.actor.log_stds.fill_(100)
    assert torch.allclose(a.actor(obs,base)[1].exp(),torch.full((4,7),10.))


def test_unit_density_invariant_under_physical_scale_and_matches_tanh():
    a=agent(); mean=torch.zeros(4,7); std=torch.zeros_like(mean)
    torch.manual_seed(21); d,p=a.actor.sample_distribution(mean,std)
    z=d/.5; raw=torch.atanh(z)
    expected=(torch.distributions.Normal(mean,torch.ones_like(mean)).log_prob(raw)-torch.log1p(-z*z)).sum(-1,keepdim=True)
    torch.testing.assert_close(p,expected,atol=2e-5,rtol=2e-5)
    a.set_action_scale(.2)
    torch.manual_seed(21); d2,p2=a.actor.sample_distribution(mean,std)
    torch.testing.assert_close(d2,d*.4);torch.testing.assert_close(p,p2,rtol=0,atol=0)
    assert a.target_entropy==-3.5


def test_mean_actor_min_target_and_temperature_gradient():
    a=agent(temperature_impl='serl_softplus')
    q1=torch.tensor([[1.]],requires_grad=True);q2=torch.tensor([[3.]],requires_grad=True)
    a.actor_q(q1,q2).backward()
    assert q1.grad.item()==q2.grad.item()==.5
    assert torch.minimum(q1,q2).item()==1
    assert abs(a.alpha.item()-1)<1e-6
    loss=a.temperature_loss(torch.tensor([[-1.]]))
    loss.backward()
    assert a.log_alpha.grad.item()>0  # entropy1 > target-3.5 -> lower alpha


def test_optimizer_warmup_counts_gradient_steps_and_checkpoint_resume(tmp_path):
    torch.set_num_threads(1)
    a=agent(optimizer_warmup_steps=2)
    before={k:v.clone() for k,v in a.actor.state_dict().items()}
    m0=a.update(batch(),update_actor=False)
    assert m0['critic_lr']==0
    m1=a.update(batch(),update_actor=False)
    assert m1['critic_lr']==pytest.approx(1.5e-4)
    assert all(torch.equal(v,before[k]) for k,v in a.actor.state_dict().items())
    assert a.actor_optimizer_steps==0 and a.critic_optimizer_steps==2
    path=tmp_path/'agent.pt';a.save(path); b=ResidualSAC.load(path)
    assert b.critic_optimizer_steps==2
    torch.manual_seed(17);m=a.update(batch())
    torch.manual_seed(17);n=b.update(batch())
    assert m==n
    assert all(torch.equal(x,y) for x,y in zip(a.q1.parameters(),b.q1.parameters()))


def test_shared_encoder_critic_adapts_actor_stops_gradient_target_updates_once():
    a=agent(shared_visual_encoder=True,visual_encoder='serl_resnet10',
            image_shape=(2,32,32,3),visual_latent_dim=8)
    assert a.actor.obs_encoder is a.q1.obs_encoder is a.q2.obs_encoder
    assert a.q1_target.obs_encoder is a.q2_target.obs_encoder
    assert a.q1_target.obs_encoder is not a.q1.obs_encoder
    visual=a.q1.obs_encoder.visual
    assert not any(p.requires_grad for p in visual.trunk.parameters())
    x=torch.randn(2,8);base=torch.zeros(2,7);im=torch.rand(2,6,32,32)
    a.actor(x,base,im)[0].sum().backward()
    assert all(p.grad is None for p in visual.parameters())
    a.q1(x,base,im).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in visual.heads.parameters())
    params=[p for g in a.q_opt.param_groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})
    head=next(visual.heads.parameters()); target=next(a.q1_target.obs_encoder.visual.heads.parameters())
    with torch.no_grad():head.fill_(1);target.zero_()
    a.update_targets()
    torch.testing.assert_close(target,torch.full_like(target,a.config.tau))


def test_auxiliary_calql_discards_offline_actor_but_preserves_critic():
    a=agent(); before={k:v.clone() for k,v in a.actor.state_dict().items()}
    a.begin_calql('auxiliary_full_action')
    assert torch.all(a.actor.scale==1)
    for _ in range(2): a.pretrain_critic_calql(batch())
    q={k:v.clone() for k,v in a.q1.state_dict().items()}
    a.finish_calql()
    assert all(torch.equal(v,before[k]) for k,v in a.actor.state_dict().items())
    assert all(torch.equal(v,q[k]) for k,v in a.q1.state_dict().items())
    assert a.actor_optimizer_steps==0
    assert a.alpha.item()==pytest.approx(1)


def test_probe_prefix_advances_base_cache_but_never_enters_replay():
    from maniskill_myws.pld.libero_runner import run_episode
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from test_pld_libero import observation, ChunkModel
    class Env:
        prompt='bowl'
        def reset(self,*,seed):
            self.i=0
            return observation(),dict(reset_hash='same',initial_state=np.zeros(2))
        def physics_state(self):return np.array([self.i])
        def step(self,a):
            self.i+=1
            return observation(),float(self.i==5),self.i==5,False,dict(success=self.i==5)
    p=ChunkedBasePolicy(ChunkModel(),replan_steps=2)
    plain,full=run_episode(Env(),p,seed=7,image_size=4)
    replay=[];called=[]
    def residual(obs,base):called.append(base.copy());return np.zeros(7,'f')
    row,tail=run_episode(Env(),p,seed=7,image_size=4,residual=residual,probe_steps=3,on_transition=replay.append)
    assert row['length']==5 and row['probe_steps']==3 and row['active_length']==2
    assert len(replay)==len(tail)==len(called)==2
    assert row['trajectory_hash']==plain['trajectory_hash']
    np.testing.assert_array_equal(tail[0]['base_action'],full[3]['base_action'])
    early,tail=run_episode(Env(),p,seed=7,image_size=4,residual=residual,probe_steps=8)
    assert early['probe_only_success'] and len(tail)==0 and early['active_length']==0


def test_scale_schedule_uses_active_steps_only():
    from maniskill_myws.pld.libero_runner import scheduled_scale, sample_probe_steps
    cfg=dict(residual_scale=.5,residual_scale_start=.2,residual_scale_warmup_steps=25000)
    assert scheduled_scale(cfg,0)==.2
    assert scheduled_scale(cfg,12500)==pytest.approx(.35)
    assert scheduled_scale(cfg,50000)==.5
    rng=np.random.default_rng(5)
    draws=[sample_probe_steps(rng,.3,220) for _ in range(1000)]
    assert min(draws)==0 and max(draws)==66
    with pytest.raises(ValueError):sample_probe_steps(rng,1.1,220)


def test_rebind_split_allows_only_heldout_seed_change(monkeypatch,tmp_path):
    import json
    from pathlib import Path
    from maniskill_myws.pld.libero_protocol import Protocol
    from maniskill_myws.pld.libero_selection import rebind_alignment_split
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    newer=dict(cfg,eval_seeds=list(range(4000,4050)))
    manifest=dict(split_hash=Protocol(cfg).split_hash,alignment_steps=3001,checkpoint_files={'unchanged':'hash'})
    old=tmp_path/'old.json';old.write_text(json.dumps(manifest))
    monkeypatch.setattr(Protocol,'require_alignment',lambda self,path:manifest)
    bound=rebind_alignment_split(old,cfg,newer)
    assert bound['checkpoint_files']==manifest['checkpoint_files']
    assert bound['split_hash']==Protocol(newer).split_hash
    assert bound['parent_alignment']['path']==str(old.resolve())
    with pytest.raises(ValueError,match='training'):
        rebind_alignment_split(old,cfg,dict(newer,train_env_seeds=[1,2]))


def test_counterfactual_replay_preserves_prefix_and_branches_only_one_action():
    from maniskill_myws.pld.libero_counterfactual import branch_rollout, ranking_summary
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from test_pld_libero import observation, ChunkModel
    class Env:
        prompt='bowl';horizon=4
        def reset(self,*,seed):
            self.i=0;self.x=0.
            return observation(),dict(reset_hash='same',initial_state=np.zeros(2))
        def physics_state(self):return np.array([self.i,self.x])
        def step(self,a):
            self.i+=1;self.x+=float(a[0]);done=self.i==4
            return observation(),float(done and self.x>0),done,False,dict(success=done and self.x>0)
    p=ChunkedBasePolicy(ChunkModel(),replan_steps=2)
    prefix=[np.full(7,.1,'f'),np.full(7,.2,'f')]
    a=branch_rollout(Env(),p,seed=8,prefix=prefix,candidate=lambda o,b:b,image_size=4)
    b=branch_rollout(Env(),p,seed=8,prefix=prefix,candidate=lambda o,b:b,image_size=4)
    assert a['trajectory_hash']==b['trajectory_hash']
    c=branch_rollout(Env(),p,seed=8,prefix=prefix,candidate=lambda o,b:np.ones(7,'f'),image_size=4)
    assert a['prefix_hash']==c['prefix_hash'] and a['branch_state']==c['branch_state']
    r=ranking_summary([dict(q=[.5,.8,.2],returns=[1.,0.,1.])])
    assert r['false_positive_rate']==1 and r['pairwise_ranking_accuracy']==0


def test_v3_training_fields_bound_into_checkpoint_regimen():
    import json
    from pathlib import Path
    from maniskill_myws.pld.libero_protocol import residual_training_spec, require_specialist_regimen
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v3.json').read_text())
    spec=residual_training_spec(cfg)
    for key in ('probe_fraction','residual_actor_impl','shared_visual_encoder','actor_q_reduction',
                'optimizer_warmup_steps','temperature_impl','calql_init','residual_density'):
        assert spec[key]==cfg[key]
        other=dict(cfg);other[key]='changed'
        with pytest.raises(ValueError,match='regimen'):
            require_specialist_regimen(dict(training_spec=spec),other,source_validation=True)


def test_serl_dense_initialization_uses_zero_bias():
    a=agent()
    for model in (a.actor,a.q1,a.q2):
        for m in model.modules():
            if isinstance(m,torch.nn.Linear):assert torch.count_nonzero(m.bias)==0


@pytest.mark.skipif(not __import__('os').environ.get('PLD_SERL_REFERENCE'),reason='pinned SERL opt-in')
def test_serl_distribution_and_shared_encoder_gradient_parity():
    import os,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(os.environ['PLD_SERL_REFERENCE'])/'serl/serl_launcher'))
    import jax,jax.numpy as jnp,flax.linen as nn
    from serl_launcher.networks.actor_critic_nets import TanhMultivariateNormalDiag,Policy,Critic,ensemblize
    from serl_launcher.networks.mlp import MLP
    from serl_launcher.common.common import ModuleDict
    from serl_launcher.common.encoding import EncodingWrapper
    from serl_launcher.vision.resnet_v1 import resnetv1_configs,PreTrainedResNetEncoder
    a=agent();mean=torch.zeros(4,7);std=torch.zeros_like(mean)
    torch.manual_seed(7);d,p=a.actor.sample_distribution(mean,std)
    ref=TanhMultivariateNormalDiag(loc=jnp.zeros((4,7)),scale_diag=jnp.ones((4,7)))
    np.testing.assert_allclose(np.asarray(ref.log_prob(jnp.array((d/.5).detach().numpy()))),p.detach().numpy()[:,0],atol=2e-5)
    trunk=resnetv1_configs['resnetv1-10-frozen'](pre_pooling=True,name='pretrained_encoder')
    vis=PreTrainedResNetEncoder(pretrained_encoder=trunk,pooling_method='spatial_learned_embeddings',
        num_spatial_blocks=8,bottleneck_dim=8)
    enc=EncodingWrapper(encoder={'image':vis},use_proprio=False,enable_stacking=True,image_keys=('image',))
    policy=Policy(encoder=enc,network=MLP((8,),activate_final=True),action_dim=7,std_parameterization='uniform')
    critic=Critic(encoder=enc,network=ensemblize(MLP,2)(hidden_dims=(8,),activate_final=True))
    module=ModuleDict({'actor':policy,'critic':critic})
    obs={'image':jnp.ones((1,1,32,32,3),dtype=jnp.uint8)};actions=jnp.zeros((1,7))
    params=module.init(jax.random.PRNGKey(0),actor=[obs],critic=[obs,actions])['params']
    ga=jax.grad(lambda p:module.apply({'params':p},obs,name='actor').mean().sum())(params)
    gc=jax.grad(lambda p:module.apply({'params':p},obs,actions,name='critic').sum())(params)
    from flax.traverse_util import flatten_dict
    flat=flatten_dict(params);af=flatten_dict(ga);cf=flatten_dict(gc)
    trunkkeys=[k for k in flat if 'pretrained_encoder' in '/'.join(k)]
    visualkeys=[k for k in flat if 'encoder' in '/'.join(k) and k not in trunkkeys]
    assert trunkkeys and visualkeys
    assert all(np.all(np.asarray(af[k])==0) and np.all(np.asarray(cf[k])==0) for k in trunkkeys)
    assert all(np.all(np.asarray(af[k])==0) for k in visualkeys)
    assert any(np.any(np.asarray(cf[k])!=0) for k in visualkeys)


def test_auxiliary_calql_uses_reference_optimizer_and_temperature_defaults():
    a=agent(temperature_impl='serl_softplus',optimizer_warmup_steps=2000)
    a.begin_calql('auxiliary_full_action')
    assert a.alpha.item()==pytest.approx(1.)
    m=a.pretrain_critic_calql(batch())
    assert m['critic_lr']==3e-4 and m['actor_lr']==1e-4
    assert type(a.actor_opt) is torch.optim.Adam and type(a.q_opt) is torch.optim.Adam
    a.finish_calql()
    assert a.critic_optimizer_steps==a.actor_optimizer_steps==0
    assert a.alpha.item()==pytest.approx(1.)
    assert type(a.q_opt) is torch.optim.AdamW


def test_source_selection_freezes_mode_across_deterministic_and_otf(tmp_path):
    import json
    from pathlib import Path
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    from maniskill_myws.pld.libero_selection import select_source_validation,require_source_selection
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v3.json').read_text())
    alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'weights')
    checkpoint.with_suffix('.json').write_text(json.dumps(dict(training_steps=100,checkpoint_sha256=file_sha256(checkpoint))))
    paths=[]
    for mode,successes,mag in [('otf',35,.1),('deterministic_actor',40,.01)]:
        p=tmp_path/mode;(p/'eval').mkdir(parents=True);paths.append(p)
        (p/'metadata.json').write_text(json.dumps(dict(status='COMPLETED')))
        (p/'config.json').write_text(json.dumps(dict(cfg,command_options=dict(alignment_manifest=str(alignment)))))
        row=dict(source=Protocol(cfg).source,target=Protocol(cfg).source,distance='D0',evaluation_scope='source_validation',
            evaluation_policy=mode,seeds=cfg['validation_env_seeds'],episodes=50,SR_residual=successes/50,
            residual_successes=successes,alignment_sha256=file_sha256(alignment),residual_checkpoint=str(checkpoint),
            checkpoint_sha256=file_sha256(checkpoint),executed_residual_mean_abs=mag)
        (p/'eval/summary.json').write_text(json.dumps([row]))
    result=select_source_validation(paths,cfg,role='residual',policy='auto')
    assert result['policy']=='deterministic_actor' and result['success_rate']==.8
    selected=tmp_path/'selection.json';selected.write_text(json.dumps(result))
    assert require_source_selection(selected,cfg,alignment,checkpoint)['policy']=='deterministic_actor'
    result['policy']='otf';selected.write_text(json.dumps(result))
    with pytest.raises(ValueError):require_source_selection(selected,cfg,alignment,checkpoint)
    # Equal performance/intervention: actual checkpoint age, not input order.
    later=tmp_path/'later.pt';later.write_bytes(b'later weights')
    later.with_suffix('.json').write_text(json.dumps(dict(training_steps=200,checkpoint_sha256=file_sha256(later))))
    first_summary=paths[0]/'eval/summary.json';row=json.loads(first_summary.read_text())[0]
    row.update(SR_residual=.8,residual_successes=40,executed_residual_mean_abs=.01,
               residual_checkpoint=str(later),checkpoint_sha256=file_sha256(later))
    first_summary.write_text(json.dumps([row]))
    assert select_source_validation(paths,cfg,role='residual',policy='auto')['checkpoint']==str(checkpoint)


def test_replay_checkpoint_retains_circular_write_position(tmp_path):
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    a=ReplayBuffer(3,1,1)
    for i in range(5):a.add([i],[i],[i],i,[i+1],[i+1],False)
    path=tmp_path/'replay.npz';a.save(path);b=ReplayBuffer(3,1,1);b.load(path)
    assert a.pos==b.pos==2
    a.add([9],[9],[9],9,[9],[9],True);b.add([9],[9],[9],9,[9],[9],True)
    np.testing.assert_array_equal(a.obs,b.obs)


def test_training_snapshot_restores_replay_and_all_rng_streams(tmp_path):
    from maniskill_myws.pld.libero_training_state import save_training_state,restore_training_state
    from maniskill_myws.pld.libero_runner import residual_policy
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    a=agent();r=residual_policy(a,dict(otf_rollout_actions=1),'otf',seed=9)
    replay=ReplayBuffer(3,8,7)
    for i in range(3):replay.add(np.full(8,i),np.zeros(7),np.zeros(7),0,np.zeros(8),np.zeros(7),False)
    probe=np.random.default_rng(19);ckpt=tmp_path/'a.pt';a.save(ckpt)
    folder=tmp_path/'snapshot'
    save_training_state(folder,ckpt,replay,r,probe,dict(episode=100,active_steps=50000))
    expected=(np.random.rand(),torch.rand(2),probe.integers(10000),replay.sample(2).obs)
    state=restore_training_state(folder,replay,r,probe)
    actual=(np.random.rand(),torch.rand(2),probe.integers(10000),replay.sample(2).obs)
    assert state['episode']==100
    for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)
    with pytest.raises(FileExistsError):save_training_state(folder,ckpt,replay,r,probe,{})


def test_train_resume_matches_uninterrupted_episode_boundary(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from maniskill_myws.pld import libero_experiment as exp
    from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    from test_pld_libero import observation,ChunkModel
    torch.set_num_threads(1)
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v3.json').read_text())
    cfg.update(device='cpu',rl_image_size=4,visual_encoder='none',calql_updates=1,calql_n_actions=2,
        batch_size=2,buffer_capacity=30,warmup_episodes=2,active_steps=6,online_steps=10,
        checkpoint_active_steps=[2,6],stage_review_active_steps=[2],warmup_review_required=False,
        optimizer_warmup_steps=2,probe_fraction=.5)
    cfg['source']['horizon']=2
    protocol=Protocol(cfg);alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    offline=ReplayBuffer(2,8,7,image_shape=(2,4,4,3));im=np.zeros((2,4,4,3),np.uint8)
    for i in range(2):offline.add(np.zeros(8),np.zeros(7),np.zeros(7),1,np.zeros(8),np.zeros(7),True,1,images=im,next_images=im)
    path=tmp_path/'offline.npz';offline.save(path,kind='aligned_base_success',source=protocol.source,
        split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,alignment_sha256=file_sha256(alignment))
    class Env:
        prompt='source';horizon=2
        def __init__(self,*a,**kw):pass
        def reset(self,seed):self.i=0;return observation(),dict(reset_hash=str(seed),initial_state=np.zeros(2))
        def physics_state(self):return np.array([self.i,0.])
        def step(self,a):self.i+=1;return observation(),float(self.i==2),self.i==2,False,dict(success=self.i==2)
        def close(self):pass
    monkeypatch.setattr(exp,'LiberoEnv',Env)
    def run(name,decision,resume=None):
        torch.manual_seed(43);np.random.seed(43)
        root=tmp_path/name
        for folder in ('logs','checkpoints'):(root/folder).mkdir(parents=True)
        r=SimpleNamespace(path=root,meta={},begin_cuda_phase=lambda:0,end_cuda_phase=lambda kind:dict(allocated_bytes=0,reserved_bytes=0))
        monkeypatch.setattr(exp,'await_stage_review',lambda *args:decision)
        exp.train(cfg,SimpleNamespace(offline_buffer=path,alignment_manifest=alignment,resume_stage=resume),r,
                  ChunkedBasePolicy(ChunkModel()),SimpleNamespace(),protocol)
        return r
    full=run('full',True)
    stopped=run('stopped',False)
    resumed=run('resumed',True,stopped.path/'stages/active_2')
    a=ResidualSAC.load(full.meta['checkpoint']);b=ResidualSAC.load(resumed.meta['checkpoint'])
    assert full.meta['active_steps']==resumed.meta['active_steps']
    for model in ('actor','q1','q2','q1_target','q2_target'):
        for x,y in zip(getattr(a,model).parameters(),getattr(b,model).parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(a.log_alpha,b.log_alpha,atol=0,rtol=0)


def test_paired_video_categories_show_rescues_and_harm():
    from maniskill_myws.pld.libero_runner import paired_outcome_category
    assert paired_outcome_category(False,True)=='rescue'
    assert paired_outcome_category(True,False)=='harm'
    assert paired_outcome_category(False,False) is None
    assert paired_outcome_category(True,True) is None


def test_offline_replay_can_load_into_compact_capacity(tmp_path):
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    a=ReplayBuffer(10,1,1);a.add([1],[0],[0],1,[2],[0],True)
    path=tmp_path/'offline.npz';a.save(path)
    b=ReplayBuffer(1,1,1);b.load(path)
    assert len(b)==1 and b.pos==0
    np.testing.assert_array_equal(b.obs,[[1]])


def test_full_replay_resize_preserves_chronological_order(tmp_path):
    from maniskill_myws.pld.replay_buffer import ReplayBuffer
    a=ReplayBuffer(3,1,1)
    for i in range(5):a.add([i],[0],[0],0,[i],[0],False)
    path=tmp_path/'full.npz';a.save(path)
    b=ReplayBuffer(5,1,1);b.load(path)
    assert b.pos==3 and not b.full
    np.testing.assert_array_equal(b.obs[:3,0],[2,3,4])
