"""Discarded real pi0+C-RADIO smoke. Failures never authorize main training."""
from pathlib import Path
import gc,hashlib,json,pickle,sys,time
import numpy as np
import torch
import torch.nn.functional as F
BASE=Path('/workspace/visual-alignment-repro-20261029');ROOT=BASE/'maniskill_myws'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/adaptation'),str(BASE/'private'),str(BASE/'pi0-d0d1-visual-alignment')]
from vram_gate import validate
from alignment_port import project,cosine_loss,middle_state,disable_extra_augmentation
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from maniskill_myws.pld.libero_openpi import load_pi0_pytorch
import jax
E=BASE/'evidence';result={'passed':False,'discarded':True,'main_training_updates':0,'required_free_headroom_GiB':4}

def save():E.joinpath('model-smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def digest(params):
 h=hashlib.sha256()
 for name,value in sorted(params.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def batch(items,start,size):
 b=jax.tree.map(lambda *values:torch.from_numpy(np.stack(values)).cuda(),*[x[0] for x in items[start:start+size]])
 actions=b.pop('actions').float();obs=Observation.from_dict(b)
 for view,key in enumerate(['base_0_rgb','left_wrist_0_rgb']):
  pixels=np.stack([x[1][view] for x in items[start:start+size]])
  target=torch.from_numpy(pixels).cuda().permute(0,3,1,2).float()/255*2-1
  assert torch.equal(obs.images[key],target),'Teacher/student pixel normalization mismatch'
 assert not bool(obs.image_masks['right_wrist_0_rgb'].any())
 return obs,actions

def run():
 result['vram_gate']=validate(E)
 torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=True
 assert torch.cuda.get_device_capability()==(8,0)
 result['torch_version']=torch.__version__;result['torch_cuda']=torch.version.cuda
 result['jax_version']=jax.__version__;result['gpu']=torch.cuda.get_device_name()
 result['cuda_total_bytes']=torch.cuda.get_device_properties(0).total_memory
 with (BASE/'private/smoke/batch.pkl').open('rb') as f:items=pickle.load(f)
 from teacher.hf_model import RADIOModel,RADIOConfig
 from safetensors.torch import load_file
 print('LOAD_C_RADIO',flush=True)
 teacher=RADIOModel(RADIOConfig.from_json_file(BASE/'private/teacher/config.json'))
 teacher.load_state_dict(load_file(BASE/'private/teacher/model.safetensors'),strict=True)
 teacher.eval().requires_grad_(False).cuda();features=[]
 for start in range(0,8,2):
  pixels=torch.from_numpy(np.stack([x[1] for x in items[start:start+2]])).reshape(4,224,224,3).permute(0,3,1,2).cuda().float()/255
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):_,grid=teacher(pixels)
  assert grid.shape==(4,196,1024)
  grid=F.interpolate(grid.float().reshape(4,14,14,1024).permute(0,3,1,2),size=(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1)
  features.append(grid.reshape(2,2,16,16,1024).cpu().half())
 targets=torch.cat(features);assert torch.isfinite(targets).all()
 result['teacher_peak_allocated_bytes']=torch.cuda.max_memory_allocated();result['teacher_frozen']=all(not p.requires_grad for p in teacher.parameters())
 del teacher,pixels,grid,features;gc.collect();torch.cuda.empty_cache()
 print('LOAD_PI0',flush=True)
 path=BASE/'private/historical/checkpoints/d0/3000/model.safetensors'
 model=load_pi0_pytorch(Pi0Config(dtype='float32'),path)
 if hasattr(model.sample_actions,'_torchdynamo_orig_callable'):model.sample_actions=model.sample_actions._torchdynamo_orig_callable
 assert all(p.dtype==torch.float32 for p in model.parameters())
 disable_extra_augmentation(model);model.gradient_checkpointing_enable();model.cuda().eval()
 result['model_parameters']=sum(p.numel() for p in model.parameters())
 # Verify block9 prefix against the exact pinned joint-forward arithmetic.
 llm=model.paligemma_with_expert.paligemma.language_model;capture={}
 def before_norm(module,args,kwargs):capture['residual']=args[0].detach()
 def after_mlp(module,args,output):capture['mlp']=output.detach()
 h1=llm.layers[8].post_attention_layernorm.register_forward_pre_hook(before_norm,with_kwargs=True)
 h2=llm.layers[8].mlp.register_forward_hook(after_mlp)
 obs,actions=batch(items,0,1)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  loss=model(obs,actions,noise=torch.zeros_like(actions),time=torch.full((1,),.5,device='cuda'))
 reference=(capture['residual']+capture['mlp'])[:,:3*256].reshape(1,3,16,16,2048)[:,:2]
 h1.remove();h2.remove()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):hidden,mask=middle_state(model,obs)
 rmse=float((hidden.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-8))
 result['middle_prefix_relative_rmse']=rmse;result['middle_prefix_tolerance']=.005
 assert rmse<.005,'Prefix-only extraction differs from full joint block9'
 assert torch.isfinite(loss).all()
 del capture,hidden,reference,loss,obs,actions;gc.collect();torch.cuda.empty_cache()
 print('PREFIX_INTEGRATION_PASSED',rmse,flush=True)
 # Five projector smoke updates, using actual frozen policy/teacher features.
 hs=[];ms=[]
 for start in range(0,8,2):
  obs,_=batch(items,start,2)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h,m=middle_state(model,obs)
  hs.append(h.cpu());ms.append(m.cpu())
 hidden=torch.cat(hs).cuda().float();mask=torch.cat(ms).cuda();target=targets.cuda().float()
 from alignment.model import projector_init
 initial=projector_init(jax.random.key(101),2048,1024)
 projector={k:torch.tensor(np.asarray(v).copy(),device='cuda',requires_grad=True) for k,v in initial.items()}
 popt=torch.optim.Adam(list(projector.values()),lr=.001);calibration=[]
 for i in range(5):
  popt.zero_grad(set_to_none=True);loss=cosine_loss(project(projector,hidden,freeze=False),target,mask)
  assert torch.isfinite(loss);loss.backward();popt.step();calibration.append(float(loss.detach()))
 for value in projector.values():value.requires_grad_(False);value.grad=None
 phash=digest(projector);result['projector_smoke_calibration']=calibration
 del popt,initial,hidden,hs,ms,target,mask,h,m,obs,loss;gc.collect();torch.cuda.empty_cache()
 np.savez(BASE/'private/smoke/projector_discarded.npz',**{k:v.cpu().numpy() for k,v in projector.items()})
 model.train().requires_grad_(True)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-5/101,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=False)
 sample=llm.layers[8].self_attn.q_proj.weight;before=sample.detach().cpu().clone()
 torch.cuda.reset_peak_memory_stats();rows=[]
 # Repeated updates catch optimizer state allocation and sustained memory behavior.
 for step in range(3):
  start_time=time.time();opt.zero_grad(set_to_none=True);flow_value=0.;align_value=0.
  for start in range(0,8,2):
   obs,actions=batch(items,start,2)
   with torch.autocast('cuda',dtype=torch.bfloat16):flow=model(obs,actions).mean()/4
   assert torch.isfinite(flow);flow.backward();flow_value+=float(flow.detach())
   del flow
   # Sequential backwards sum the identical objective gradient with lower graph retention.
   with torch.autocast('cuda',dtype=torch.bfloat16):hidden,mask=middle_state(model,obs)
   align=cosine_loss(project(projector,hidden),targets[start:start+2].cuda(),mask)
   assert torch.isfinite(align);(.2*align/4).backward();align_value+=float(align.detach())/4
   del hidden,mask,align,obs,actions
  grads=[p for p in model.parameters() if p.grad is not None]
  assert grads and all(bool(torch.isfinite(p.grad).all()) for p in grads)
  norm=float(torch.nn.utils.clip_grad_norm_(grads,1.));assert np.isfinite(norm)
  opt.step();torch.cuda.synchronize()
  free,total=torch.cuda.mem_get_info();row=dict(step=step+1,L_flow=flow_value,L_align=align_value,grad_norm=norm,seconds=time.time()-start_time,
   allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=free,total_bytes=total)
  rows.append(row);result['updates']=rows;save();print('MODEL_SMOKE_UPDATE',json.dumps(row),flush=True)
  # Reserve measured driver-free memory as well as allocator-headroom margin.
  if min(free,total-torch.cuda.max_memory_reserved())<4*(1<<30):raise RuntimeError('Model/batch lack the preregistered4GiB free VRAM margin')
 assert not torch.equal(sample.detach().cpu(),before),'Model optimizer did not change active weights'
 assert digest(projector)==phash and all(v.grad is None for v in projector.values())
 assert all(p.dtype==torch.float32 for p in model.parameters())
 for state in opt.state.values():
  assert all(v.dtype==torch.float32 for v in state.values() if isinstance(v,torch.Tensor))
 result.update(passed=True,completed_discarded_updates=3,effective_batch=8,microbatch=2,accumulation=4,projector_frozen=True,fp32_master_and_adam_states=True,
  minimum_headroom_bytes=min(min(r['free_bytes'],r['total_bytes']-r['peak_reserved_bytes']) for r in rows))
 save();print('MODEL_ALIGNMENT_UPDATE_SMOKE_PASSED',flush=True)

try:run()
except BaseException as exc:
 result['error_type']=type(exc).__name__;result['error']=str(exc);save();raise
