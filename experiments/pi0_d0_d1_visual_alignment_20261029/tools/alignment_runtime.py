"""Deterministic source K4 pixels, checked teacher features and frozen-projector loss."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import torch
import jax
from common import WORK,ROOT,EXPERIMENT,SFT,BATCH,sha,write
BASE=WORK.parent
SOURCE=BASE/'pi0-d0d1-visual-alignment'
sys.path.insert(0,str(SOURCE))
from alignment.data import VIEWS,augmentation_params,replay,image_hash
from alignment.cache import CacheReader
from openpi_client.image_tools import resize_with_pad
from openpi.models.model import Observation
from alignment_port import middle_state,project,cosine_loss,disable_extra_augmentation
ALIGN=WORK/'alignment'

def corpus_root():
 root=ALIGN/'corpus';(root/'configs').mkdir(parents=True,exist_ok=True);(root/'private').mkdir(exist_ok=True)
 native=root/'private/native'
 if not native.exists():native.symlink_to(BASE/'private/native',target_is_directory=True)
 inputs=json.loads((BASE/'evidence/smoke-input-receipts.json').read_text())
 manifest={}
 from common import TASKS
 for task in ['D0','D1']:
  path='libero_spatial/'+TASKS[task]['name']+'_demo.hdf5'
  item=next(r for r in inputs if r['repo']=='yifengzhu-hf/LIBERO-datasets' and r['path']==path)
  manifest[task]=dict(path=path,sha256=item['sha256'],revision=item['revision'])
 p=root/'configs/dataset_manifest.json'
 if p.exists():assert json.loads(p.read_text())==manifest,'Dataset binding changed'
 else:write(p,manifest)
 return root

def augmented(ds,index,position):
 raw=ds.raw(int(index));episode,frame=ds.index[int(index)]
 sample=f'{ds.task}/{ds.audit["demo_names"][episode]}/{frame}';aug=int(position%4)
 params={};images=[]
 for key,view in zip(['observation/image','observation/wrist_image'],VIEWS):
  params[view]=augmentation_params(sample,aug,view)
  image=replay(resize_with_pad(raw[key],224,224),params[view]);raw[key]=image;images.append(image)
 return ds.transform(raw),dict(sample_id=sample,augmentation=aug,params=params,image_sha256=[image_hash(x) for x in images])

class Objective:
 def __init__(self,model,stage,*,projector_required=True):
  self.stage=stage;self.active=stage in ('d0','full');self.model=model
  disable_extra_augmentation(model)
  self.task=SFT[stage]['task'];self.reader=CacheReader(ALIGN/'cache'/self.task)
  assert self.reader.manifest['metadata']['tasks']==[self.task]
  self.projector=None;self.lambda_align=0.;self.last_identities=[]
  self.projector_path=ALIGN/'projector.npz'
  if projector_required:
   decision=json.loads((ALIGN/'lambda_decision.json').read_text());cal=json.loads((ALIGN/'projector_calibration.json').read_text())
   assert decision['tasks']==['D0'] and cal['tasks']==['D0'] and cal['updates']==200 and cal['policy_parameters_unchanged']
   assert sha(self.projector_path)==cal['projector_sha256']
   self.binding=dict(teacher_repo='nvidia/C-RADIOv3-L',teacher_revision='9d0413465e8a91e67bbf2c1ad342815478d1b906',
    projector_sha256=cal['projector_sha256'],lambda_decision_sha256=sha(ALIGN/'lambda_decision.json'),cache_manifest_sha256=sha(ALIGN/'cache'/self.task/'manifest.json'),
    data_tasks=[self.task],calibration_tasks=['D0'],K=4,augmentation_selection='(update*effective_batch+sample_slot)%4; original sample permutation unchanged',
    student_block_index=8,cameras=list(VIEWS),teacher_frozen=True,projector_frozen=True,
    method='active_spatial_regularizer' if self.active else 'inherited_aligned_D0_only')
   if self.active:
    with np.load(self.projector_path) as f:self.projector={k:torch.from_numpy(f[k].copy()).cuda() for k in f.files}
    self.lambda_align=float(decision['selected']);assert np.isfinite(self.lambda_align) and self.lambda_align>0
  else:self.binding={}
 def batch(self,ds,indices,step,offset,device='cuda'):
  items=[];features=[];identities=[]
  for slot,index in enumerate(indices):
   row,identity=augmented(ds,index,step*BATCH+offset+slot)
   cached=self.reader.lookup(identity['sample_id'],identity['augmentation'])
   if cached['image_sha256']!=identity['image_sha256'] or cached['params']!=identity['params']:
    raise ValueError('Teacher/student pixel replay mismatch')
   items.append(row);identities.append(identity)
   if self.active:features.append(cached['features'])
  data=jax.tree.map(lambda *v:torch.as_tensor(np.stack(v),device=device),*items)
  actions=data.pop('actions').float();obs=Observation.from_dict(data)
  target=torch.from_numpy(np.stack(features)).to(device) if features else None
  self.last_identities=identities
  return obs,actions,target
 def loss(self,observation,target):
  if not self.active:raise ValueError('Expert scope has no effective prefix alignment objective')
  with torch.autocast('cuda',dtype=torch.bfloat16):hidden,mask=middle_state(self.model,observation)
  return cosine_loss(project(self.projector,hidden),target,mask)
 def save_metadata(self,target):
  data=dict(self.binding,lambda_align=self.lambda_align)
  write(Path(target)/'visual_alignment.json',data)
  if self.projector is not None:
   assert all(not p.requires_grad and p.grad is None for p in self.projector.values())
   actual={k:v.detach().cpu().numpy() for k,v in self.projector.items()}
   with np.load(self.projector_path) as f:
    assert all(np.array_equal(actual[k],f[k]) for k in actual),'Frozen projector changed'
  shutil.copy2(self.projector_path,Path(target)/'projector.npz')
  return data

def make_objective(model,stage):return Objective(model,stage)
