"""Student-facing, idempotent exploration of the three published policies."""
import gc
from common import *
from evaluate import FlowModel,PairedVideoEnv,verified_episode

class SeminarPolicy:
 def __init__(self,method):
  import torch
  from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
  self.method=method;self.selection=json.loads((WORK/'final_selection.json').read_text())[method]
  self.checkpoint=Path(self.selection['checkpoint']);self.norm=sha(NORM)
  norm=self.checkpoint/'assets/d0/norm_stats.json';assert sha(norm)==self.norm
  weight=self.checkpoint/('residual.pt' if method=='residual' else 'model.safetensors');assert sha(weight)==self.selection['weights_sha256']
  if method=='residual':assert sha(WORK/'sft/d0/3000/model.safetensors')==self.selection['base_weights_sha256']
  self.model=load_model(WORK/'sft/d0/3000' if method=='residual' else self.checkpoint).to('cuda')
  self.agent=None
  if method=='residual':
   from maniskill_myws.pld.sac import ResidualSAC
   self.agent=ResidualSAC.load(weight,device='cuda')
  self.env=None
 def scene(self,task='D1',seed=30000):
  from maniskill_myws.pld.libero_backend import ChunkedBasePolicy,convert_observation
  if self.env:self.env.close()
  self.task,self.seed=task,seed;self.env=PairedVideoEnv(task,'final')
  self.raw,self.info=self.env.reset(seed=seed);self.flow=FlowModel(self.model,task);self.base=ChunkedBasePolicy(self.flow,replan_steps=5);self.base.reset(seed)
  obs=convert_observation(self.raw,self.env.prompt)
  return dict(prompt=self.env.prompt,state=obs['state'],images=obs['images'],reset_hash=self.info['reset_hash'],initial_observation_hash=self.env.initial_observation_hash)
 def action(self):
  from maniskill_myws.pld.libero_backend import convert_observation
  from maniskill_myws.pld.libero_runner import actor_residual
  self.base.reset(self.seed)  # Re-running this cell recomputes the same first action.
  base=self.base.act(self.raw);unit=np.zeros(7,np.float32)
  if self.agent:
   residual=actor_residual(self.agent)
   unit=residual(convert_observation(self.raw,self.env.prompt),base)
  action=np.clip(base+2*unit,-1,1)
  recorded=verified_episode(WORK/'final'/self.method/self.task/str(self.seed),self.method,self.task,self.seed,self.checkpoint)
  np.testing.assert_allclose(action,recorded['first_action'],atol=1e-6)
  return dict(base=base,unit_residual=unit,executed_action=action,checked_against_recorded_first_action=True)
 def rollout(self):
  """Reuse the one verified final episode; do not append a new observation."""
  return verified_episode(WORK/'final'/self.method/self.task/str(self.seed),self.method,self.task,self.seed,self.checkpoint)
 def video(self,row):
  from IPython.display import Video
  return Video(str(WORK/'final'/self.method/self.task/str(self.seed)/row['video']),embed=True,width=720)
 def success_rates(self):
  stats=json.loads((EXPERIMENT/'results/success_rates.json').read_text())
  return {task:stats[self.method+'_'+task] for task in TASKS}
 def close(self):
  import torch
  if self.env:self.env.close();self.env=None
  self.model.to('cpu');self.model=None;self.agent=None;gc.collect();torch.cuda.empty_cache()
