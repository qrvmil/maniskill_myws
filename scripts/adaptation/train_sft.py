"""Independent full or action-expert SFT with exact update counts and resumable RNG."""
import argparse, math, shutil, time
from common import *

def lr(step,total):
 peak=3e-5;end=3e-6;warm=100
 if step<warm:return peak/(warm+1)+(peak-peak/(warm+1))*step/warm
 return end+(peak-end)*.5*(1+math.cos(math.pi*min(1,(step-warm)/(total-warm))))

def save(model,opt,stage,step,scope_rows,indices,initial_hash):
 import torch,safetensors.torch
 from storage import publish,push_progress
 root=WORK/'sft'/stage;target=root/str(step);tmp=root/(str(step)+'.partial')
 if target.exists():return target
 tmp.mkdir(parents=True,exist_ok=True)
 for name in ('scope.json','gradient_scope.json'):
  if (root/name).exists():shutil.copy2(root/name,tmp/name)
 np.save(tmp/'sample_indices.npy',indices)
 safetensors.torch.save_model(model,str(tmp/'model.safetensors'))
 torch.save(dict(optimizer=opt.state_dict(),step=step,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state()),tmp/'training_state.pt')
 (tmp/'assets/d0').mkdir(parents=True,exist_ok=True);shutil.copy2(NORM,tmp/'assets/d0/norm_stats.json')
 assert sha(tmp/'assets/d0/norm_stats.json')==sha(NORM)
 frozen={r['name']:r['sha256'] for r in scope_rows if not r['trainable']}
 changed=[]
 for n,p in model.named_parameters():
  if n in frozen and tensor_hash(p)!=frozen[n]:changed.append(n)
 if changed:raise ValueError('Frozen parameter mutation: '+str(changed))
 write(tmp/'training.json',dict(stage=stage,completed_updates=step,source_task=SFT[stage]['task'],training_seed=SFT[stage]['seed'],
  effective_batch=BATCH,microbatch=MICROBATCH,accumulation=BATCH//MICROBATCH,
  initial_weights_sha256=initial_hash,weights_sha256=sha(tmp/'model.safetensors'),normalization_sha256=sha(NORM),
  frozen_scope_verified=True,frozen_tensors=len(frozen),scope=SFT[stage]['scope'],
  total_parameters=sum(r['numel'] for r in scope_rows),trainable_parameters=sum(r['numel'] for r in scope_rows if r['trainable']),
  optimizer='AdamW float32 master parameters/states',lr_schedule='cosine warmup100 3e-5 to3e-6',final_budget=SFT[stage]['updates']))
 tmp.replace(target)
 write(root/'latest.json',dict(checkpoint=str(target),step=step))
 receipt=publish(target,f'checkpoints/{stage}/{step}',full_readback=step==SFT[stage]['updates'],defer=True)
 write(WORK/'evidence'/f'sft_{stage}_{step}.json',dict(checkpoint=str(target),**json.loads((target/'training.json').read_text()),hf_commit=receipt['commit']))
 metrics=root/'metrics.jsonl'
 if metrics.exists():
  dest=ROOT/'experiments/pi0_d0_d1_three_adaptations/evidence'/f'{stage}_metrics.jsonl'
  shutil.copy2(metrics,dest)
 push_progress(f'exp: save {stage} SFT after {step} updates queued for verified HF publication')
 # Older local optimizer states are unnecessary once a newer resumable save has been read back.
 for old in root.iterdir():
  if old.is_dir() and old.name.isdecimal() and int(old.name)<step:
   state=old/'training_state.pt'
   if state.exists() and (old/'upload_receipt.json').exists():state.unlink()
 return target

def train(stage,*,profile=False):
 import torch,jax,safetensors.torch
 from storage import publish,push_progress
 torch.set_num_threads(8);torch.manual_seed(SFT[stage]['seed']);np.random.seed(SFT[stage]['seed'])
 torch.backends.cuda.matmul.allow_tf32=True
 ds=NativeDataset(SFT[stage]['task'])
 root=WORK/'sft'/stage;root.mkdir(parents=True,exist_ok=True)
 initial=ASSETS/'pi0_base_torch' if stage=='d0' else WORK/'sft/d0/3000'
 initial_hash=sha(initial/'model.safetensors')
 checkpoint=initial;step=0
 if not profile and (root/'latest.json').exists():
  latest=json.loads((root/'latest.json').read_text());checkpoint=Path(latest['checkpoint']);step=latest['step']
 if step and not (checkpoint/'upload_receipt.json').exists():
  publish(checkpoint,f'checkpoints/{stage}/{step}',full_readback=step==SFT[stage]['updates'],defer=True)
 if step==SFT[stage]['updates']:
  write(WORK/'evidence'/f'{stage}_complete.json',dict(stage=stage,updates=step,resumed_completed=True))
  return
 model=load_model(checkpoint,training=True)
 # Master weights stay float32; freezing never changes their stored dtype or bytes.
 assert all(p.dtype==torch.float32 for p in model.parameters())
 rows=scope(model,SFT[stage]['scope'])
 write(root/'scope.json',dict(parameters=rows,total_parameters=sum(r['numel'] for r in rows),trainable_parameters=sum(r['numel'] for r in rows if r['trainable'])))
 shutil.copy2(root/'scope.json',WORK/'evidence'/f'sft_{stage}_scope.json')
 assert not any('lora' in r['name'].lower() for r in rows)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
 if step:
  state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
  opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng']);np.random.set_state(state['numpy_rng'])
 model.to('cuda')
 for opt_state in opt.state.values():
  for k,v in opt_state.items():
   if isinstance(v,torch.Tensor) and k!='step':opt_state[k]=v.to('cuda')
 rng=np.random.default_rng(SFT[stage]['seed']);n=SFT[stage]['updates']*BATCH
 indices=np.concatenate([rng.permutation(len(ds)) for _ in range(math.ceil(n/len(ds)))])[:n].reshape(-1,BATCH)
 np.save(root/'sample_indices.npy',indices)
 last=time.time();start=step;peak=0
 for current in range(step,SFT[stage]['updates'] if not profile else step+1):
  opt.zero_grad(set_to_none=True)
  for group in opt.param_groups:group['lr']=lr(current,SFT[stage]['updates'])
  total_loss=0.
  for offset in range(0,BATCH,MICROBATCH):
   obs,actions=batch(ds,indices[current,offset:offset+MICROBATCH],'cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(obs,actions).mean()/(BATCH//MICROBATCH)
   if not torch.isfinite(loss):raise FloatingPointError('SFT loss nonfinite')
   loss.backward();total_loss+=float(loss.detach())
   del obs,actions,loss
  if current==0:
   gradients={n:dict(gradient_present=p.grad is not None,gradient_finite=p.grad is None or bool(torch.isfinite(p.grad).all())) for n,p in model.named_parameters()}
   write(root/'gradient_scope.json',gradients)
   shutil.copy2(root/'gradient_scope.json',WORK/'evidence'/f'sft_{stage}_gradient_scope.json')
   assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
   for prefix in ('paligemma_with_expert.gemma_expert.model.','state_proj.','action_in_proj.','action_out_proj.','action_time_mlp_in.','action_time_mlp_out.'):
    assert any(n.startswith(prefix) and p.grad is not None and bool(torch.any(p.grad!=0)) for n,p in model.named_parameters()),prefix
  grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.))
  if not np.isfinite(grad):raise FloatingPointError('Nonfinite SFT grad')
  opt.step();torch.cuda.synchronize();peak=max(peak,torch.cuda.max_memory_allocated())
  row=dict(step=current+1,loss=total_loss,lr=lr(current,SFT[stage]['updates']),grad_norm=grad,seconds=time.time()-last,cuda_peak_allocated=peak)
  last=time.time();print('SFT',stage,json.dumps(row),flush=True)
  if profile:
   from evaluate import FlowModel,PairedVideoEnv
   from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
   env=PairedVideoEnv('D0','profile',video=False);raw,_=env.reset(seed=23010)
   model.eval();flow=FlowModel(model,'D0');policy=ChunkedBasePolicy(flow,replan_steps=5);policy.reset(23010)
   action=policy.act(raw);assert action.shape==(7,) and np.isfinite(action).all()
   env.step(action);env.close()
   write(WORK/'evidence/memory_profile.json',dict(stage=stage,**row,effective_batch=BATCH,microbatch=MICROBATCH,fp32_master=True,profile_discarded=True))
   push_progress('exp: verify full-model memory and inference before fixed-budget D0 SFT')
   return
  with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  step=current+1
  interval=1000 if stage=='d0' else 500
  teacher_candidate=stage=='full' and step%250==0 and not (WORK/'teacher.json').exists()
  if step%interval==0 or step==SFT[stage]['updates'] or teacher_candidate:
   target=save(model,opt,stage,step,rows,indices,initial_hash)
   if stage=='full' and step<1500 and not (WORK/'teacher.json').exists():
    from evaluate import validation_teacher
    result=validation_teacher(model,target)
    write(WORK/'evidence'/f'teacher_validation_{step}.json',result)
    if result['successes']>0:
     teacher=WORK/'teacher_policy';teacher.mkdir(exist_ok=True)
     import os
     os.link(target/'model.safetensors',teacher/'model.safetensors')
     shutil.copytree(target/'assets',teacher/'assets',dirs_exist_ok=True)
     shutil.copy2(target/'training.json',teacher/'training.json')
     write(teacher/'validation.json',result)
     write(WORK/'teacher.json',dict(checkpoint=str(teacher),source_checkpoint=str(target),step=step,validation=result))
     publish(teacher,'checkpoints/teacher',full_readback=True,defer=True)
    model.train()
   # Bound local storage: remote history is retained; keep common D0, teacher and latest.
   teacher=json.loads((WORK/'teacher.json').read_text())['checkpoint'] if (WORK/'teacher.json').exists() else None
   for old in root.iterdir():
    if old.is_dir() and old.name.isdecimal() and int(old.name)<step and str(old)!=teacher and (old/'upload_receipt.json').exists():
     shutil.rmtree(old)
 write(WORK/'evidence'/f'{stage}_complete.json',dict(stage=stage,updates=step,cuda_peak_allocated=peak))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=SFT);p.add_argument('--profile',action='store_true');a=p.parse_args();train(a.stage,profile=a.profile)
