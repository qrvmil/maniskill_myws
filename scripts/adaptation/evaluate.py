"""Paired generated resets, explicit flow noise and one verified video per final episode."""
import argparse, contextlib, time
from common import *

class FlowModel:
 def __init__(self,model,task):
  import torch
  from openpi.policies.policy import Policy
  from openpi import transforms as T
  self.model=model;self.task=task
  cfg=config(task);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
  self.policy=Policy(model,transforms=[*dc.data_transforms.inputs,T.Normalize(dc.norm_stats,use_quantiles=False),*dc.model_transforms.inputs],
    output_transforms=[*dc.model_transforms.outputs,T.Unnormalize(dc.norm_stats,use_quantiles=False),*dc.data_transforms.outputs],
    is_pytorch=True,pytorch_device='cuda')
  self.reset(0)
 def reset(self,seed):self.rng=np.random.default_rng(seed);self.noise_hashes=[]
 def infer(self,raw):
  import torch
  from maniskill_myws.pld.libero_backend import openpi_observation
  noise=self.rng.standard_normal((50,32)).astype(np.float32)
  self.noise_hashes.append(hashlib.sha256(noise.tobytes()).hexdigest())
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   return self.policy.infer(openpi_observation(raw,TASKS[self.task]['name'].replace('_',' ')),noise=noise)

class PairedVideoEnv:
 def __init__(self,task,role,*,video=True):
  from maniskill_myws.pld.libero_backend import LiberoEnv
  self._env=LiberoEnv(TASKS[task],render_size=256);self.task=task;self.role=role;self.video=video;self.frames=[]
  assert self._env.prompt==TASKS[task]['name'].replace('_',' ')
 def __getattr__(self,name):return getattr(self._env,name)
 def record(self,raw):
  if self.video:
   self.frames.append(np.concatenate([np.ascontiguousarray(raw[k][::-1,::-1]) for k in ('agentview_image','robot0_eye_in_hand_image')],axis=1))
 def reset(self,*,seed):
  path=WORK/'reset_states'/self.role/self.task/f'{seed}.npz'
  if path.exists():
   old=np.load(path,allow_pickle=False)
   raw,info=self._env.reset(seed=seed,initial_state=old['sampled_initial_state'])
   assert info['reset_hash']==str(old['reset_hash']),f'Reset drift {self.task}/{seed}'
   assert np.array_equal(info['initial_state'],old['initial_state'])
  else:
   raw,info=self._env.reset(seed=seed);path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(path,initial_state=info['initial_state'],sampled_initial_state=info['sampled_initial_state'],reset_hash=info['reset_hash'],seed=seed,task=self.task)
  self.frames=[];self.record(raw);return raw,info
 def step(self,action):
  raw,*rest=self._env.step(action);self.record(raw);return (raw,*rest)
 def close(self):self._env.close()

def validation_teacher(model,checkpoint):
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import run_episode
 states=[p.requires_grad for p in model.parameters()];model.eval()
 flow=FlowModel(model,'D1');base=ChunkedBasePolicy(flow,replan_steps=5)
 env=PairedVideoEnv('D1','validation',video=False);rows=[]
 try:
  for seed in range(22000,22010):
   row,_=run_episode(env,base,seed=seed)
   row.pop('physics_states',None);row.pop('initial_state',None);rows.append(row)
   print('TEACHER_VALIDATION',checkpoint,seed,row['success'],flush=True)
 finally:
  env.close()
  for p,requires in zip(model.parameters(),states):p.requires_grad_(requires)
 return dict(checkpoint=str(checkpoint),n=len(rows),successes=sum(r['success'] for r in rows),episodes=rows,seeds=list(range(22000,22010)))

def episode(env,base,*,seed,method,task,checkpoint,residual=None,scale=2.):
 from maniskill_myws.pld.libero_runner import run_episode
 from maniskill_myws.pld.libero_sanity_videos import validate_video
 import imageio.v2 as imageio
 row,transitions=run_episode(env,base,seed=seed,residual=residual,residual_scale=scale,image_size=128,record_progress=True)
 row.pop('physics_states',None);row.pop('initial_state',None)
 outcome='success' if row['success'] else 'failure'
 folder=WORK/'final'/method/task/str(seed);folder.mkdir(parents=True,exist_ok=True)
 path=folder/f'{method}_{task}_seed{seed}_{outcome}.mp4'
 assert len(env.frames)==row['length']+1
 imageio.mimwrite(path,env.frames,fps=20,codec='libx264',quality=7,macro_block_size=1)
 verified=validate_video(path,row['length']+1)
 row.update(method=method,task=task,checkpoint=str(checkpoint),normalization_sha256=sha(NORM),
   first_action=transitions[0]['action'].tolist(),first_base_action=transitions[0]['base_action'].tolist(),
   video=path.name,video_sha256=sha(path),video_verification=verified,includes_terminal_frame=True,
   cameras=['agentview','robot0_eye_in_hand'],camera_render_size=256,replan_steps=5,horizon=220,
   flow_noise_hashes=base.model.noise_hashes)
 write(folder/'episode.json',row)
 return row

def verified_episode(folder,method,task,seed,checkpoint):
 from maniskill_myws.pld.libero_sanity_videos import validate_video
 path=folder/'episode.json'
 if not path.exists():return None
 row=json.loads(path.read_text())
 assert (row['method'],row['task'],row['seed'],row['checkpoint'])==(method,task,seed,str(checkpoint))
 assert row['normalization_sha256']==sha(NORM)
 reset=np.load(WORK/'reset_states/final'/task/f'{seed}.npz');assert row['reset_hash']==str(reset['reset_hash'])
 video=folder/row['video'];assert sha(video)==row['video_sha256'];validate_video(video,row['length']+1)
 return row

def final(method):
 import torch
 from storage import publish,push_progress
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import residual_policy
 selection=json.loads((WORK/'final_selection.json').read_text())
 if method=='residual':assert sha(WORK/'sft/d0/3000/model.safetensors')==selection[method]['base_weights_sha256']
 checkpoint=Path(selection[method]['checkpoint']);assert sha(checkpoint/('residual.pt' if method=='residual' else 'model.safetensors'))==selection[method]['weights_sha256']
 torch.set_num_threads(4);torch.manual_seed(0)
 model=load_model(WORK/'sft/d0/3000' if method=='residual' else checkpoint).to('cuda')
 agent=None
 if method=='residual':
  from maniskill_myws.pld.sac import ResidualSAC
  agent=ResidualSAC.load(checkpoint/'residual.pt',device='cuda')
 for task in TASKS:
  flow=FlowModel(model,task);base=ChunkedBasePolicy(flow,replan_steps=5);env=PairedVideoEnv(task,'final')
  rows=[]
  try:
   for seed in range(30000,30050):
    folder=WORK/'final'/method/task/str(seed)
    row=verified_episode(folder,method,task,seed,checkpoint)
    if row is None:
     residual=None if agent is None else residual_policy(agent,{'otf_rollout_actions':1},'deterministic_actor',seed=seed+9000000)
     row=episode(env,base,seed=seed,method=method,task=task,checkpoint=checkpoint,residual=residual)
    if not (folder/'upload_receipt.json').exists():
     publish(folder,f'final/{method}/{task}/{seed}',full_readback=True)
    rows.append(row);write(WORK/'final'/method/task/'episodes.json',rows)
    print('FINAL_EPISODE',method,task,seed,row['success'],row['length'],flush=True)
    if len(rows)%10==0:
     from report import interim
     interim();push_progress(f'exp: preserve final {method}/{task}, {len(rows)} of 50 paired episodes')
  finally:env.close()
 publish(WORK/'reset_states','reset_states',full_readback=True)
 write(WORK/'evidence'/f'final_{method}_complete.json',dict(method=method,episodes=100))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method',choices=['full','expert','residual']);args=p.parse_args();final(args.method)
