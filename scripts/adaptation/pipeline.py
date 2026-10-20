"""Serial GPU continuation; every phase is resumable and final results never select weights."""
import os,subprocess,sys,time
from common import *

def call(script,*args):
 print('PIPELINE_STAGE',script,args,flush=True)
 subprocess.run([sys.executable,'-u',str(ROOT/'scripts/adaptation'/script),*args],cwd=ROOT,check=True)

def main():
 while not (WORK/'evidence/d0_complete.json').exists():
  status=subprocess.check_output(['supervisorctl','status','pi0_three_d0'],text=True).split()[1]
  if status in ('EXITED','FATAL','STOPPED'):raise RuntimeError('D0 did not complete; inspect pi0_three_d0 logs')
  time.sleep(10)
 while subprocess.check_output(['supervisorctl','status','pi0_three_d0'],text=True).split()[1] in ('RUNNING','STOPPING'):time.sleep(2)
 call('train_sft.py','full')
 call('train_sft.py','expert')
 if not (WORK/'teacher.json').exists():raise RuntimeError('No registered early full-SFT teacher has nonzero validation SR; residual warmup cannot run')
 if not (ASSETS/'serl_resnet10.pt').exists():
  subprocess.run([sys.executable,str(ROOT/'scripts/pld/prepare_serl_encoder.py'),'--output',str(ASSETS/'serl_resnet10.pt')],cwd=ROOT,check=True)
 call('train_residual.py','--zero-check')
 call('offline_replay.py')
 call('train_residual.py')
 from storage import publish,push_progress
 selected={}
 for method in ('full','expert','residual'):
  path=WORK/'rl/final' if method=='residual' else WORK/'sft'/method/'1500'
  weights=path/('residual.pt' if method=='residual' else 'model.safetensors')
  assert sha(path/'assets/d0/norm_stats.json')==sha(NORM)
  selected[method]=dict(checkpoint=str(path),weights_sha256=sha(weights),normalization_sha256=sha(NORM),selection='fixed preregistered last budget; no final evaluations used',hf_repo=HF_REPO)
 selected['residual']['base_weights_sha256']=sha(WORK/'sft/d0/3000/model.safetensors')
 if (WORK/'final_selection.json').exists():assert selected==json.loads((WORK/'final_selection.json').read_text())
 else:write(WORK/'final_selection.json',selected)
 binding=WORK/'selection';binding.mkdir(exist_ok=True)
 write(binding/'final_selection.json',selected)
 write(ROOT/'experiments/pi0_d0_d1_three_adaptations/final_selection.json',selected)
 publish(binding,'selection',full_readback=True);push_progress('exp: freeze three final checkpoints before opening final evaluation seeds')
 for method in ('full','expert','residual'):call('evaluate.py',method)
 call('report.py')
 call('build_notebook.py')
 call('publish_card.py')
 write(WORK/'complete.json',dict(objective_complete=True,episodes=300,videos=300,executed_notebook=True,github_branch='exp/pi0-d0-d1-three-adaptations-20261020',hf_repo=HF_REPO))
 print('THREE_ADAPTATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
