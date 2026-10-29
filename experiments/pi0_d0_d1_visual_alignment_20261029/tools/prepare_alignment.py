"""Seen-only teacher cache, D0 projector calibration and discarded lambda pilot."""
from pathlib import Path
import argparse,gc,json,math,sys,time
import numpy as np
import jax
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'scripts/adaptation'),str(Path(__file__).parent)]
from common import *
from alignment_runtime import ALIGN,BASE,SOURCE,corpus_root,Objective
sys.path[:0]=[str(BASE/'private'),str(SOURCE)]
from alignment.model import projector_init
from alignment.stats import choose_lambda
from alignment.cache import CacheWriter
from alignment.corpus import Corpus
from alignment_port import middle_state,project,cosine_loss,disable_extra_augmentation

def policy_digest(model):
 import hashlib
 h=hashlib.sha256()
 for name,p in model.named_parameters():h.update(name.encode());h.update(bytes.fromhex(tensor_hash(p)))
 return h.hexdigest()

def gate():
 assert json.loads((BASE/'evidence/vram-gate.json').read_text())['passed']
 smoke=json.loads((BASE/'evidence/model-smoke.json').read_text());assert smoke['passed'] and smoke['effective_batch']==BATCH and smoke['microbatch']==MICROBATCH
 assert (BASE/'evidence/model-preflight.exitcode').read_text().strip()=='0'
 assert json.loads((WORK/'evidence/simulator_smoke.json').read_text())['passed']
 import subprocess
 ecc=subprocess.check_output(['nvidia-smi','--query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True).strip()
 assert [int(x.strip()) for x in ecc.split(',')]==[0,0],'GPU ECC counter changed; long training gate closed'
 assert sha(NORM)=='a06c2e745bf91d8dd6a3fa6377560d3104e6cad621da4ecebe1372c74bbb388e'

def order(ds,updates):
 rng=np.random.default_rng(0);n=updates*BATCH
 return np.concatenate([rng.permutation(len(ds)) for _ in range(math.ceil(n/len(ds)))])[:n].reshape(updates,BATCH)

def cache(task):
 target=ALIGN/'cache'/task
 if (target/'manifest.json').exists():
  from alignment.cache import CacheReader
  CacheReader(target);print('CACHE_ALREADY_VERIFIED',task,flush=True);return
 assert not target.exists(),'Incomplete cache retained; inspect/archive it before rebuilding'
 if task=='D1':assert (WORK/'evidence/d0_complete.json').exists(),'D1 cache is built only after common D0 is fixed'
 corpus=Corpus(corpus_root(),tasks=(task,))
 from teacher.hf_model import RADIOModel,RADIOConfig
 from safetensors.torch import load_file
 teacher=RADIOModel(RADIOConfig.from_json_file(BASE/'private/teacher/config.json'))
 teacher.load_state_dict(load_file(BASE/'private/teacher/model.safetensors'),strict=True)
 teacher.eval().requires_grad_(False).cuda();before=policy_digest(teacher)
 writer=CacheWriter(target,shard_size=32,metadata=dict(tasks=[task],K=4,teacher_repo='nvidia/C-RADIOv3-L',teacher_revision='9d0413465e8a91e67bbf2c1ad342815478d1b906',
  dataset_manifest_sha256=sha(corpus_root()/'configs/dataset_manifest.json'),teacher_frozen=True,teacher_input='same uint8 RGB224/255 as replayed by student; internal RADIO conditioner',grid_resize='bilinear align_corners=False14x14→16x16'))
 from torch.nn.functional import interpolate
 started=time.time()
 for i,row in enumerate(corpus.records):
  items=[corpus.augmented(row,aug) for aug in range(4)]
  pixels=torch.from_numpy(np.stack([x['images'] for x in items])).reshape(8,224,224,3).permute(0,3,1,2).cuda().float()/255
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):_,features=teacher(pixels)
  assert features.shape==(8,196,1024)
  grid=interpolate(features.float().reshape(8,14,14,1024).permute(0,3,1,2),size=(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(4,2,16,16,1024).cpu().numpy().astype(np.float16)
  for aug,item in enumerate(items):writer.add(row['sample_id'],aug,grid[aug],item['params'],item['image_sha256'])
  if i%100==0:print('TEACHER_CACHE',task,i,len(corpus.records),round(time.time()-started,1),flush=True)
 manifest=writer.finish();assert policy_digest(teacher)==before
 write(ALIGN/f'cache_{task}.json',dict(tasks=[task],samples=len(corpus.records),entries=len(manifest['index']),bytes=sum(s['bytes'] for s in manifest['shards']),manifest_sha256=sha(target/'manifest.json'),teacher_unchanged=True,seconds=time.time()-started))
 write(WORK/'evidence'/f'alignment_cache_{task}.json',json.loads((ALIGN/f'cache_{task}.json').read_text()))
 from storage import publish,push_progress
 metadata=ALIGN/'published_cache_metadata'/task;metadata.mkdir(parents=True)
 import shutil
 shutil.copy2(target/'manifest.json',metadata/'manifest.json');shutil.copy2(ALIGN/f'cache_{task}.json',metadata/'summary.json')
 publish(metadata,'alignment/cache_metadata/'+task,full_readback=True)
 push_progress('exp: verify frozen C-RADIO '+task+' cache and exact K4 pixel identities')


def calibrate():
 out=ALIGN/'projector.npz'
 if out.exists():
  m=json.loads((ALIGN/'projector_calibration.json').read_text());assert m['updates']==200 and sha(out)==m['projector_sha256'];return
 assert not (ALIGN/'projector_calibration.json').exists()
 torch.manual_seed(0);model=load_model(ASSETS/'pi0_base_torch').cuda();model.requires_grad_(False).eval();disable_extra_augmentation(model)
 before=policy_digest(model);ds=NativeDataset('D0');objective=Objective(model,'d0',projector_required=False)
 params={k:torch.tensor(np.asarray(v).copy(),device='cuda',requires_grad=True) for k,v in projector_init(jax.random.key(101),2048,1024).items()}
 optimizer=torch.optim.Adam(list(params.values()),lr=.001);rows=[];indices=order(ds,200)
 for step in range(200):
  optimizer.zero_grad(set_to_none=True);value=0.
  for offset in range(0,BATCH,MICROBATCH):
   obs,_,target=objective.batch(ds,indices[step,offset:offset+MICROBATCH],step,offset)
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):hidden,mask=middle_state(model,obs)
   loss=cosine_loss(project(params,hidden.detach(),freeze=False),target,mask)/(BATCH//MICROBATCH)
   assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
  optimizer.step();rows.append(dict(step=step+1,loss=value))
  if (step+1)%10==0:print('PROJECTOR_CALIBRATION',step+1,value,flush=True)
 assert policy_digest(model)==before
 np.savez(out,**{k:p.detach().cpu().numpy() for k,p in params.items()})
 data=dict(tasks=['D0'],updates=200,seed=101,lr=.001,optimizer='Adam(.9,.999,eps1e-8)',policy_parameters_unchanged=True,policy_parameter_digest=before,initial_weights_sha256=sha(ASSETS/'pi0_base_torch/model.safetensors'),projector_sha256=sha(out),losses=rows)
 write(ALIGN/'projector_calibration.json',data);write(WORK/'evidence/projector_calibration.json',data)
 from storage import publish,push_progress
 public=ALIGN/'projector_release';public.mkdir(exist_ok=True)
 import shutil
 shutil.copy2(out,public/'projector.npz');shutil.copy2(ALIGN/'projector_calibration.json',public/'calibration.json')
 publish(public,'alignment/projector',full_readback=True);push_progress('exp: freeze D0-only projector after200calibration updates')


def norm(model):return math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None))

def pilot():
 if (ALIGN/'lambda_decision.json').exists():return
 cal=json.loads((ALIGN/'projector_calibration.json').read_text());assert cal['updates']==200 and cal['tasks']==['D0']
 torch.manual_seed(0);np.random.seed(0);model=load_model(ASSETS/'pi0_base_torch',training=True).cuda();model.requires_grad_(True)
 ds=NativeDataset('D0');objective=Objective(model,'d0',projector_required=False)
 with np.load(ALIGN/'projector.npz') as f:objective.projector={k:torch.from_numpy(f[k].copy()).cuda() for k in f.files}
 objective.lambda_align=.2
 opt=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
 from train_sft import lr
 rows=[];indices=order(ds,20)
 for step in range(20):
  for group in opt.param_groups:group['lr']=lr(step,3000)
  opt.zero_grad(set_to_none=True);flow_value=0.;align_value=0.
  for offset in range(0,BATCH,MICROBATCH):
   obs,actions,_=objective.batch(ds,indices[step,offset:offset+MICROBATCH],step,offset)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(obs,actions).mean()/(BATCH//MICROBATCH)
   assert torch.isfinite(loss);loss.backward();flow_value+=float(loss.detach())
  fn=norm(model);flow_grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None};opt.zero_grad(set_to_none=True)
  for offset in range(0,BATCH,MICROBATCH):
   obs,_,target=objective.batch(ds,indices[step,offset:offset+MICROBATCH],step,offset)
   loss=objective.loss(obs,target)/(BATCH//MICROBATCH);assert torch.isfinite(loss);loss.backward();align_value+=float(loss.detach())
  an=norm(model);r=.2*an/max(fn,1e-30)
  if not np.isfinite(r) or r<=0:raise ValueError('Alignment gradient absent or nonfinite')
  for n,p in model.named_parameters():
   if p.grad is not None:p.grad.mul_(.2)
   if n in flow_grads:
    v=flow_grads.pop(n).cuda()
    if p.grad is None:p.grad=v
    else:p.grad.add_(v)
  del flow_grads
  gradient=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None],1.));assert np.isfinite(gradient)
  opt.step();torch.cuda.synchronize()
  row=dict(step=step+1,L_flow=flow_value,L_align=align_value,flow_grad_norm=fn,align_grad_norm=an,weighted_gradient_ratio=r)
  rows.append(row);print('D0_LAMBDA_PILOT',json.dumps(row),flush=True)
 decision=dict(tasks=['D0'],pilot_updates=20,initial=.2,selected=choose_lambda([r['weighted_gradient_ratio'] for r in rows]),median_last10=float(np.median([r['weighted_gradient_ratio'] for r in rows[-10:]])),rule='r>1:.2/r;r<.01:min(2,.02/r);otherwise.2;single adjustment;fixed for D0 and D1-full',pilot_weights_discarded=True,projector_sha256=sha(ALIGN/'projector.npz'))
 write(ALIGN/'pilot.json',rows);write(ALIGN/'lambda_decision.json',decision);write(WORK/'evidence/lambda_decision.json',decision)
 from storage import publish,push_progress
 release=ALIGN/'pilot_release';release.mkdir(exist_ok=True)
 import shutil
 for name in ['pilot.json','lambda_decision.json']:shutil.copy2(ALIGN/name,release/name)
 publish(release,'alignment/pilot',full_readback=True);push_progress('exp: select lambda on D0 gradients before fresh main initialization')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','calibrate','pilot']);p.add_argument('task',nargs='?',choices=['D0','D1']);args=p.parse_args()
 torch.set_num_threads(8);gate()
 cache(args.task) if args.mode=='cache' else calibrate() if args.mode=='calibrate' else pilot()
