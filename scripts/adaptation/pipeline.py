"""Serial GPU continuation; every phase is resumable and final results never select weights."""
import os,subprocess,sys,time
from common import *

def call(script,*args):
 print('PIPELINE_STAGE',script,args,flush=True)
 subprocess.run([sys.executable,'-u',str(ROOT/'scripts/adaptation'/script),*args],cwd=ROOT,check=True)

def service_status(name):
 # supervisorctl status returns code3 for a healthy completed one-shot service.
 result=subprocess.run(['supervisorctl','status',name],text=True,capture_output=True,check=False)
 fields=result.stdout.split()
 if len(fields)<2:raise RuntimeError('Cannot read supervisor status for '+name+': '+result.stderr)
 return fields[1]

def barrier():
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
  call('publication_queue.py','drain')
  subprocess.run(['/venv/main/bin/python','-u',str(EXPERIMENT/'tools/save_progress.py'),'exp: verify prior phase logs before next major stage'],cwd=ROOT,check=True)

def main():
 d0_service=os.environ.get('PI0_D0_SERVICE','pi0_three_d0')
 while not (WORK/'evidence/d0_complete.json').exists():
  status=service_status(d0_service)
  if status in ('EXITED','FATAL','STOPPED'):raise RuntimeError('D0 did not complete; inspect pi0_three_d0 logs')
  time.sleep(10)
 while service_status(d0_service) in ('RUNNING','STOPPING'):time.sleep(2)
 barrier()
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
  subprocess.run([sys.executable,'-u',str(EXPERIMENT/'tools/prepare_alignment.py'),'cache','D1'],cwd=ROOT,check=True)
 call('train_sft.py','full')
 barrier()
 call('train_sft.py','expert')
 barrier()
 if not (WORK/'teacher.json').exists():raise RuntimeError('No registered early full-SFT teacher has nonzero validation SR; residual warmup cannot run')
 call('revalidate_teacher.py')
 if not (ASSETS/'serl_resnet10.pt').exists():
  subprocess.run([sys.executable,str(ROOT/'scripts/pld/prepare_serl_encoder.py'),'--output',str(ASSETS/'serl_resnet10.pt')],cwd=ROOT,check=True)
 call('train_residual.py','--zero-check')
 call('offline_replay.py')
 call('train_residual.py')
 barrier()
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':call('audit_online.py','final')
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
 write(EXPERIMENT/'final_selection.json',selected)
 publish(binding,'selection',full_readback=True);push_progress('exp: freeze three final checkpoints before opening final evaluation seeds')
 barrier()
 for method in ('full','expert','residual'):call('evaluate.py',method)
 call('publication_queue.py','drain')
 call('report.py')
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
  sys.path.insert(0,str(EXPERIMENT/'tools'))
  from alignment_report import publish_card
  publish_card()
 else:
  call('build_notebook.py')
  call('publish_card.py')
 write(WORK/'complete.json',dict(objective_complete=True,episodes=300,videos=300,executed_notebook=os.environ.get('PI0_VISUAL_ALIGNMENT')!='1',github_branch=os.environ.get('PI0_GITHUB_BRANCH','exp/pi0-d0-d1-three-adaptations-20261020'),hf_repo=HF_REPO))
 print('THREE_ADAPTATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
