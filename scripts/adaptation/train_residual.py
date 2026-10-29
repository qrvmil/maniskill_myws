"""Corrected V3 learner with V4 D0/D1 roles, teacher warmup and potential shaping."""
import argparse, shutil, time
from common import *
from shaping import ShapedEnv,returns

SCALE=2.
ACTIVE_STEPS=50000
WARMUP_EPISODES=20
SHAPE=(2,128,128,3)

def model_signature(model):return {n:tensor_hash(p) for n,p in model.named_parameters()}

class TeacherResidual:
 """Runs an independent teacher cache; the runner still advances the D0 base cache."""
 def __init__(self,teacher,env):self.teacher=teacher;self.env=env;self.records=[]
 def reset(self,seed):self.teacher.reset(seed);self.records=[]
 def __call__(self,obs,base):
  action=self.teacher.act(self.env.raw)
  unit=(action-base)/SCALE
  assert np.max(np.abs(unit))<=1
  executed=np.clip(base+SCALE*unit,-1,1)
  np.testing.assert_allclose(executed,action,atol=2e-7)
  self.records.append(dict(teacher_action=action.tolist(),d0_base_action=base.tolist(),executed_action=executed.tolist(),unit_residual=unit.tolist(),reconstruction_max_error=float(np.max(np.abs(executed-action)))))
  return unit

def create_agent():
 from maniskill_myws.pld.sac import SACConfig,ResidualSAC
 cfg=SACConfig(8,7,action_scale=SCALE,visual_encoder='serl_resnet10',image_shape=SHAPE,
  calql_n_actions=10,otf_backup_actions=1,target_entropy=-3.5,otf_include_base_action=True,otf_backup_entropy=False,
  calql_alpha=5.,calql_temp=1.,calql_importance_sample=True,calql_max_target_backup=True,calql_backup_entropy=False,
  residual_actor_impl='serl_uniform_std',residual_density='unit',shared_visual_encoder=True,actor_q_reduction='mean',
  optimizer_warmup_steps=2000,temperature_impl='serl_softplus',weight_decay=.01)
 agent=ResidualSAC(cfg,device='cuda')
 path=ASSETS/'serl_resnet10.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['converted_sha256']
 agent.load_visual_encoder(path)
 return agent

def zero_check():
 import torch
 from evaluate import FlowModel,PairedVideoEnv
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import run_episode
 from storage import publish,push_progress
 root=WORK/'zero_residual';root.mkdir(exist_ok=True)
 if (root/'complete.json').exists():
  assert json.loads((root/'complete.json').read_text())['base_weights_sha256']==sha(WORK/'sft/d0/3000/model.safetensors')
  return
 model=load_model(WORK/'sft/d0/3000').to('cuda');rows=[]
 for task in TASKS:
  flow=FlowModel(model,task);base=ChunkedBasePolicy(flow,replan_steps=5);env=PairedVideoEnv(task,'sanity',video=False)
  try:
   for seed in range(23000,23003):
    a,_=run_episode(env,base,seed=seed,residual_scale=SCALE)
    noises=list(flow.noise_hashes)
    b,_=run_episode(env,base,seed=seed,residual=lambda o,a:np.zeros(7,np.float32),residual_scale=SCALE)
    for k in ('reset_hash','trajectory_hash','image_hash','length','success'):assert a[k]==b[k],(task,seed,k)
    assert noises==flow.noise_hashes
    rows.append(dict(task=task,seed=seed,passed=True,**{k:a[k] for k in ('reset_hash','trajectory_hash','image_hash','length','success')}))
  finally:env.close()
 write(root/'complete.json',dict(passed=True,episodes=rows,base_weights_sha256=sha(WORK/'sft/d0/3000/model.safetensors')))
 publish(root,'zero_residual',full_readback=True);push_progress('exp: verify frozen D0 equals D0 plus zero residual on both tasks')

def reconcile_residual_checkpoint(root,base_hash,offline_hash,teacher_info):
 """Recover the highest complete episode-boundary snapshot before model loading."""
 import torch
 from storage import publish
 from publication_queue import drain
 candidates=[]
 for folder in Path(root).iterdir():
  if not folder.is_dir() or folder.name.endswith('.partial'):continue
  if folder.name not in ('after_calql','after_teacher_warmup','final') and not folder.name.startswith('active_'):continue
  meta_path=folder/'training.json'
  if not meta_path.exists():raise ValueError('Committed residual checkpoint lacks metadata: '+str(folder))
  meta=json.loads(meta_path.read_text());c=meta['counters']
  rank=(int(c['active_steps']),int(c['environment_steps']),int(c['episode']),folder.name=='final')
  candidates.append((rank,folder,meta))
 if not candidates:return None
 _,folder,meta=max(candidates,key=lambda item:item[0]);c=meta['counters']
 required=['residual.pt','training_state.pt','online_replay.npz','assets/d0/norm_stats.json']
 if c['environment_steps']:required.append('collection_traces.tar.gz')
 if not all((folder/name).is_file() for name in required):raise ValueError('Highest committed residual snapshot is incomplete')
 if not (meta['base_weights_sha256']==base_hash and meta['offline_replay_sha256']==offline_hash and meta['normalization_sha256']==sha(NORM) and sha(folder/'assets/d0/norm_stats.json')==sha(NORM)):
  raise ValueError('Residual base/data/normalization lineage changed')
 if not (meta['teacher']['step']==teacher_info['step'] and meta['teacher']['validation']['weights_sha256']==teacher_info['validation']['weights_sha256']):raise ValueError('Residual teacher lineage changed')
 if not (meta['gamma']==.99 and meta['beta']==.1 and meta['residual_scale']==SCALE and meta['frozen_d0_weights_unchanged']):raise ValueError('Residual algorithm binding changed')
 for filename,key in [('residual.pt','weights_sha256'),('training_state.pt','training_state_sha256'),('online_replay.npz','online_replay_sha256')]:
  if sha(folder/filename)!=meta[key]:raise ValueError('Residual checkpoint checksum mismatch: '+filename)
 state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
 if state['counters']!=c or state['calql_done'] is not True:raise ValueError('Residual optimizer/state counters disagree with committed metadata')
 if not (0<=c['active_steps']<=50000 and c['environment_steps']>=c['active_steps'] and c['episode']>=0):raise ValueError('Invalid residual budget counters')
 if folder.name=='final' and c['active_steps']!=50000:raise ValueError('Final residual budget incomplete')
 if folder.name=='after_calql' and any(c.values()):raise ValueError('CalQL checkpoint has online transitions')
 if folder.name=='after_teacher_warmup' and (c['episode']!=20 or c['active_steps']):raise ValueError('Warmup checkpoint counters changed')
 if folder.name.startswith('active_') and int(folder.name.split('_')[-1])!=c['active_steps']:raise ValueError('Residual checkpoint label/counters mismatch')
 result=dict(checkpoint=str(folder),counters=c);write(Path(root)/'latest.json',result)
 if not (folder/'upload_receipt.json').exists():publish(folder,'checkpoints/residual/'+folder.name,full_readback=folder.name=='final',defer=True)
 drain()
 if not (folder/'upload_receipt.json').exists():raise ValueError('Recovered residual checkpoint is not remotely verified')
 return result

def main():
 import torch
 from evaluate import FlowModel,PairedVideoEnv
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import run_episode,residual_policy
 from maniskill_myws.pld.replay_buffer import ReplayBuffer,sample_offline_online
 from maniskill_myws.pld.sac import ResidualSAC
 from storage import publish,push_progress
 torch.set_num_threads(4);torch.manual_seed(2);np.random.seed(2)
 root=WORK/'rl';root.mkdir(exist_ok=True)
 offline_info=json.loads((WORK/'offline/complete.json').read_text())
 assert offline_info['replay_sha256']==sha(WORK/'offline/replay.npz')
 offline=ReplayBuffer(offline_info['transitions'],8,7,image_shape=SHAPE);offline.load(WORK/'offline/replay.npz')
 teacher_info=json.loads((WORK/'teacher.json').read_text());assert teacher_info['validation']['successes']>0
 assert teacher_info['canonical_validation']['successes']>0
 base_path=WORK/'sft/d0/3000';base_hash=sha(base_path/'model.safetensors');assert base_hash==offline_info['base_weights_sha256']
 recovered=None
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
  recovered=reconcile_residual_checkpoint(root,base_hash,offline_info['replay_sha256'],teacher_info)
  if recovered and Path(recovered['checkpoint']).name=='final':
   final=Path(recovered['checkpoint']);c=recovered['counters']
   write(WORK/'evidence/residual_complete.json',dict(checkpoint=str(final),active_steps=c['active_steps'],environment_steps=c['environment_steps'],episodes=c['episode'],weights_sha256=sha(final/'residual.pt'),resumed_completed=True))
   return
 model=load_model(base_path).to('cuda');frozen=model_signature(model)
 assert not any(p.requires_grad for p in model.parameters())
 flow=FlowModel(model,'D1');base=ChunkedBasePolicy(flow,replan_steps=5)
 teacher_model=load_model(teacher_info['checkpoint']).to('cuda');teacher_flow=FlowModel(teacher_model,'D1');teacher=ChunkedBasePolicy(teacher_flow,replan_steps=5)
 online=ReplayBuffer(70000,8,7,image_shape=SHAPE)
 counters=dict(episode=0,active_steps=0,environment_steps=0);rows=[];calql_done=False
 last=recovered
 if last is None and (root/'latest.json').exists():last=json.loads((root/'latest.json').read_text())
 if last:
  folder=Path(last['checkpoint'])
  if not (folder/'upload_receipt.json').exists():publish(folder,'checkpoints/residual/'+folder.name,full_readback=folder.name=='final',defer=True)
  metadata=json.loads((folder/'training.json').read_text());assert metadata['base_weights_sha256']==base_hash and metadata['normalization_sha256']==sha(NORM)
  agent=ResidualSAC.load(folder/'residual.pt',device='cuda')
  s=torch.load(folder/'training_state.pt',weights_only=False,map_location='cpu');counters=s['counters'];rows=s['rows'];calql_done=s['calql_done']
  online.load(folder/'online_replay.npz');torch.set_rng_state(s['cpu_rng']);torch.cuda.set_rng_state(s['cuda_rng']);np.random.set_state(s['numpy_rng'])
 else:agent=create_agent()
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1' and counters['episode']>=WARMUP_EPISODES:
  teacher_model.to('cpu');torch.cuda.empty_cache()
 deployment=None
 metrics_path=root/'updates.jsonl'
 def update(kind,batch):
  start=time.time();metrics=agent.pretrain_critic_calql(batch) if kind=='calql' else agent.update(batch,update_actor=True)
  if not all(np.isfinite(float(v)) for v in metrics.values()):raise FloatingPointError(metrics)
  row=dict(kind=kind,environment_steps=counters['environment_steps'],active_steps=counters['active_steps'],episode=counters['episode'],seconds=time.time()-start,**metrics)
  with metrics_path.open('a') as f:f.write(json.dumps(row)+'\n')
  if agent.total_updates%100==0:print('RL_UPDATE',json.dumps(row),flush=True)
 def save(label):
  folder=root/label
  if folder.exists():
   if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':raise RuntimeError('Residual checkpoint already committed; restart to reconcile authoritative state')
   return folder
  tmp=root/(label+'.partial');tmp.mkdir(exist_ok=True)
  agent.save(tmp/'residual.pt');online.save(tmp/'online_replay.npz',base_weights_sha256=base_hash,gamma=.99,beta=.1)
  torch.save(dict(counters=counters,rows=rows,calql_done=calql_done,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),policy_rng=None if deployment is None else dict(cpu=deployment.cpu_rng,gpu=deployment.gpu_rng)),tmp/'training_state.pt')
  (tmp/'assets/d0').mkdir(parents=True,exist_ok=True);shutil.copy2(NORM,tmp/'assets/d0/norm_stats.json')
  write(tmp/'training.json',dict(counters=counters,base_checkpoint=str(base_path),base_weights_sha256=base_hash,teacher=teacher_info,
    normalization_sha256=sha(NORM),offline_replay_sha256=offline_info['replay_sha256'],gamma=.99,beta=.1,residual_scale=SCALE,
    v4_changes=['early full-SFT teacher warmup, actor/critic/temperature updates','potential-based shaping with absorbing terminal Phi=0'],
    online_collection='OTF',final_policy='deterministic_actor',probe_fraction=0,selection='fixed 50000 active transitions',
    frozen_d0_weights_unchanged=model_signature(model)==frozen))
  if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
   metadata=json.loads((tmp/'training.json').read_text())
   metadata.update(weights_sha256=sha(tmp/'residual.pt'),training_state_sha256=sha(tmp/'training_state.pt'),online_replay_sha256=sha(tmp/'online_replay.npz'))
   write(tmp/'training.json',metadata)
  assert json.loads((tmp/'training.json').read_text())['frozen_d0_weights_unchanged']
  for name in ('critic_after_calql.json','critic_after_warmup.json'):
   if (root/name).exists():shutil.copy2(root/name,tmp/name)
  if (root/'collection').exists():
   import tarfile
   with tarfile.open(tmp/'collection_traces.tar.gz','w:gz') as archive:archive.add(root/'collection',arcname='collection')
  tmp.replace(folder);write(root/'latest.json',dict(checkpoint=str(folder),counters=counters))
  publish(folder,'checkpoints/residual/'+label,full_readback=label=='final',defer=True)
  write(WORK/'evidence'/('rl_'+label+'.json'),json.loads((folder/'training.json').read_text()))
  shutil.copy2(metrics_path,EXPERIMENT/'evidence/rl_updates.jsonl')
  push_progress('exp: preserve residual '+label+' including optimizer, RNG and replay')
  # Keep only the latest large local replay; uploaded versions remain resumable on HF.
  for old in root.iterdir():
   if old.is_dir() and old!=folder and not old.name.endswith('.partial') and (old/'upload_receipt.json').exists():
    for name in ('online_replay.npz','training_state.pt'):
     if (old/name).exists():(old/name).unlink()
  return folder
 if not calql_done:
  agent.begin_calql('auxiliary_full_action')
  for _ in range(1000):update('calql',offline.sample(256))
  agent.finish_calql();calql_done=True
  from maniskill_myws.pld.libero_diagnostics import diagnose_critic
  write(root/'critic_after_calql.json',diagnose_critic(agent,offline))
  save('after_calql')
  if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
   from publication_queue import drain
   drain()
 env=ShapedEnv(PairedVideoEnv('D1','rl',video=False))
 teacher_residual=TeacherResidual(teacher,env)
 deployment=residual_policy(agent,{'otf_rollout_actions':1},'otf',seed=2000000)
 if last:
  state=torch.load(Path(last['checkpoint'])/'training_state.pt',map_location='cpu',weights_only=False)
  s=state.get('policy_rng')
  if s:deployment.cpu_rng=s['cpu'];deployment.gpu_rng=s['gpu']
 last_saved_active=(counters['active_steps']//5000)*5000
 actor_before={n:tensor_hash(p) for n,p in agent.actor.named_parameters() if not n.startswith('obs_encoder.')};alpha_before=tensor_hash(agent.log_alpha)
 critic_before={n:tensor_hash(p) for n,p in agent.q1.named_parameters() if p.requires_grad}
 try:
  while counters['episode']<WARMUP_EPISODES or counters['active_steps']<ACTIVE_STEPS:
   warmup=counters['episode']<WARMUP_EPISODES;seed=20000+counters['episode']%1000
   # Exact final active budget: shorten only the final collection episode; final evaluation always has horizon220.
   env._env._env.horizon=220 if warmup else min(220,ACTIVE_STEPS-counters['active_steps'])
   teacher_residual.reset(seed);start_pos=online.pos
   def on_transition(tr):
    online.add(**tr);counters['environment_steps']+=1;counters['active_steps']+=int(not warmup)
    update('online',sample_offline_online(offline,online,256,offline_fraction=.5))
   row,transitions=run_episode(env,base,seed=seed,residual=teacher_residual if warmup else deployment,
      residual_scale=SCALE,on_transition=on_transition,record_progress=True)
   # MC labels become available at episode completion. SAC updates during the episode use TD rewards, not MC labels.
   mc=returns(transitions)
   for i,value in enumerate(mc):online.mc_returns[(start_pos+i)%online.capacity]=value
   for t,e in zip(transitions,env.reward_evidence):assert abs(t['reward']-e['reward'])<1e-8
   row.pop('physics_states',None);row.pop('initial_state',None);row.update(warmup=warmup,**counters)
   counters['episode']+=1;rows.append(row);write(root/'episodes.json',rows)
   trace=root/'collection'/f'{counters["episode"]:04d}.json'
   write(trace,dict(seed=seed,warmup=warmup,teacher_actions=teacher_residual.records if warmup else None,
     d0_flow_noise_hashes=flow.noise_hashes,teacher_flow_noise_hashes=teacher_flow.noise_hashes if warmup else None,rewards=env.reward_evidence,mc_returns=mc.tolist()))
   print('RL_EPISODE',json.dumps(row),flush=True)
   if counters['episode']==WARMUP_EPISODES:
    actor_changed=any(tensor_hash(p)!=actor_before[n] for n,p in agent.actor.named_parameters() if n in actor_before)
    alpha_changed=tensor_hash(agent.log_alpha)!=alpha_before
    critic_changed=any(tensor_hash(p)!=critic_before[n] for n,p in agent.q1.named_parameters() if n in critic_before)
    assert actor_changed and alpha_changed and critic_changed and agent.critic_optimizer_steps>0
    write(WORK/'evidence/warmup_updates.json',dict(actor_changed=actor_changed,temperature_changed=alpha_changed,critic_changed=critic_changed,critic_optimizer_steps=agent.critic_optimizer_steps,
      actor_optimizer_steps=agent.actor_optimizer_steps,teacher_checkpoint=teacher_info['checkpoint'],base_checkpoint=str(base_path),base_unchanged=model_signature(model)==frozen))
    from maniskill_myws.pld.libero_diagnostics import diagnose_critic
    write(root/'critic_after_warmup.json',diagnose_critic(agent,offline))
    save('after_teacher_warmup')
    if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
     from publication_queue import drain
     drain()
    # Teacher has no role after warmup and can leave GPU memory.
    teacher_model.to('cpu');torch.cuda.empty_cache()
   if counters['active_steps']-last_saved_active>=5000:
    folder=save(f'active_{counters["active_steps"]}');last_saved_active=counters['active_steps']

  final=save('final')
  write(WORK/'evidence/residual_complete.json',dict(checkpoint=str(final),active_steps=counters['active_steps'],environment_steps=counters['environment_steps'],episodes=counters['episode'],weights_sha256=sha(final/'residual.pt')))
 finally:env.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--zero-check',action='store_true');args=p.parse_args()
 zero_check() if args.zero_check else main()
