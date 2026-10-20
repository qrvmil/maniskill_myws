"""Shared immutable experiment bindings and official OpenPI transforms."""
import dataclasses, hashlib, json, os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/workspace/pi0_three')
ASSETS=WORK/'assets'
HF_REPO='qrvmil/pi0-d0-d1-three-adaptations-20261020'
TASKS=json.loads((ROOT/'configs/pld_libero/multitask_sft.json').read_text())['tasks']
TASKS={k:TASKS[k] for k in ('D0','D1')}
NORM=ASSETS/'normalization'/'d0'/'norm_stats.json'
SFT={'d0':dict(task='D0',updates=3000,seed=0,scope='full'), 'full':dict(task='D1',updates=1500,seed=1,scope='full'), 'expert':dict(task='D1',updates=1500,seed=1,scope='expert')}
MICROBATCH=2
BATCH=8

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def source(task):
 return ASSETS/'native'/'libero_spatial'/(TASKS[task]['name']+'_demo.hdf5')

def config(task='D0'):
 from openpi.training import config as oc
 from openpi.models.pi0_config import Pi0Config
 cfg=oc.get_config('pi0_libero')
 return dataclasses.replace(cfg,name='pi0_three',exp_name=task,model=Pi0Config(dtype='float32'),
   data=dataclasses.replace(cfg.data,repo_id='local/pi0_three_'+task,extra_delta_transform=False,
    assets=oc.AssetsConfig(assets_dir=str(NORM.parent.parent),asset_id='d0')),
   batch_size=BATCH,num_workers=0,ema_decay=None,wandb_enabled=False)

class NativeDataset:
 """All native demos; next-action alignment and repeated-last chunk padding."""
 def __init__(self,task,*,transformed=True):
  import h5py
  from maniskill_myws.pld.multitask_data import audit_h5
  self.audit=audit_h5(source(task),task);self.episodes=[];self.index=[];self.task=task
  with h5py.File(source(task),'r') as f:
   for name in self.audit['demo_names']:
    g=f['data'][name];o=g['obs'];n=len(g['actions'])-1
    self.episodes.append(dict(image=o['agentview_rgb'][:n],wrist=o['eye_in_hand_rgb'][:n],
      state=np.concatenate([o['ee_pos'][:n],o['ee_ori'][:n],o['gripper_states'][:n]],axis=1).astype(np.float32),
      actions=g['actions'][1:].astype(np.float32)))
    self.index.extend((len(self.episodes)-1,i) for i in range(n))
  self.transform=None
  if transformed:
   from openpi import transforms as T
   dc=config(task).data.create(config(task).assets_dirs,config(task).model)
   self.transform=T.compose([*dc.data_transforms.inputs,T.Normalize(dc.norm_stats,use_quantiles=False),*dc.model_transforms.inputs])
 def __len__(self):return len(self.index)
 def raw(self,index):
  e,i=self.index[index];g=self.episodes[e]
  actions=g['actions'][np.minimum(np.arange(i,i+50),len(g['actions'])-1)]
  return {'observation/image':np.ascontiguousarray(g['image'][i][::-1,::-1]),
   'observation/wrist_image':np.ascontiguousarray(g['wrist'][i][::-1,::-1]),
   'observation/state':g['state'][i].copy(),'actions':actions.copy(),'prompt':TASKS[self.task]['name'].replace('_',' ')}
 def __getitem__(self,index):
  raw=self.raw(index);return self.transform(raw) if self.transform else raw

def batch(dataset,indices,device):
 import torch,jax
 from openpi.models.model import Observation
 rows=[dataset[int(i)] for i in indices]
 b=jax.tree.map(lambda *v:torch.as_tensor(np.stack(v)).to(device),*rows)
 actions=b.pop('actions');return Observation.from_dict(b),actions

def load_model(path,*,training=False):
 import torch
 from maniskill_myws.pld.libero_openpi import load_pi0_pytorch
 from openpi.models.pi0_config import Pi0Config
 model=load_pi0_pytorch(Pi0Config(dtype='float32'),Path(path)/'model.safetensors')
 # Avoid constructor's implicit expensive compilation. Compile may be enabled for inference after validation.
 if hasattr(model.sample_actions,'_torchdynamo_orig_callable'):
  model.sample_actions=model.sample_actions._torchdynamo_orig_callable
 if training:model.gradient_checkpointing_enable();model.train()
 else:
  model.eval();model.requires_grad_(False)
 return model

def tensor_hash(tensor):
 import torch
 t=tensor.detach().cpu().contiguous()
 return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()

def scope(model,mode):
 allow=('paligemma_with_expert.gemma_expert.','state_proj.','action_in_proj.','action_out_proj.','action_time_mlp_in.','action_time_mlp_out.')
 for name,p in model.named_parameters():p.requires_grad_(mode=='full' or name.startswith(allow))
 rows=[dict(name=n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype),trainable=p.requires_grad,sha256=tensor_hash(p)) for n,p in model.named_parameters()]
 if mode=='expert':
  assert any(r['trainable'] for r in rows)
  assert all(not r['trainable'] for r in rows if r['name'].startswith('paligemma_with_expert.paligemma.'))
 return rows
