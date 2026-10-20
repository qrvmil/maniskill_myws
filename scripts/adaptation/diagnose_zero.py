"""Diagnose base/zero equivalence before RL or any final evaluation seed."""
from common import *
def main(role):
 import torch
 from evaluate import FlowModel,PairedVideoEnv
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import run_episode
 torch.set_num_threads(4)
 model=load_model(WORK/'sft/d0/3000').to('cuda')
 flow=FlowModel(model,'D0');base=ChunkedBasePolicy(flow,replan_steps=5)
 env=PairedVideoEnv('D0',role,video=False);runs=[];rows=[]
 try:
  for index in range(4):
   row,tr=run_episode(env,base,seed=23000,residual=None if index%2==0 else lambda o,a:np.zeros(7,np.float32),residual_scale=2.)
   runs.append((row,tr,list(flow.noise_hashes)))
   compact={k:row[k] for k in ('success','length','reset_hash','trajectory_hash','image_hash','inference_seconds','wall_seconds')}
   rows.append(compact);print('DIAGNOSTIC_RUN',index,json.dumps(compact),flush=True)
 finally:env.close()
 comparisons=[]
 for i in range(3):
  a,ta,na=runs[i];b,tb,nb=runs[i+1];item=dict(before=i,after=i+1,noise_equal=na==nb,fields={})
  for field in ('obs','action','base_action','images'):
   errors=[float(np.max(np.abs(np.asarray(x[field],np.float64)-np.asarray(y[field],np.float64)))) for x,y in zip(ta,tb)]
   item['fields'][field]=dict(first_different_step=next((k for k,e in enumerate(errors) if e),None),maximum_absolute_error=max(errors))
  physics=[float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(a['physics_states'],b['physics_states'])]
  item['physics']=dict(first_different_step=next((k for k,e in enumerate(physics) if e),None),maximum_absolute_error=max(physics))
  comparisons.append(item);print('DIAGNOSTIC_COMPARISON',json.dumps(item),flush=True)
 write(WORK/'evidence'/(role+'.json'),dict(runs=rows,comparisons=comparisons,final_seeds_used=False))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--role',default='zero_diagnostic_canonical');main(p.parse_args().role)
