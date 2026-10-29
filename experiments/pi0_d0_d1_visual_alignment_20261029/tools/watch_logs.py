"""Publish explicit generated logs/metrics every five minutes; no training control."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
from common import WORK,EXPERIMENT
while True:
 result=subprocess.run(['/venv/main/bin/python','-u',str(EXPERIMENT/'tools/save_progress.py'),'exp: preserve current experiment logs and metrics'],cwd=ROOT)
 if result.returncode:print('LOG_PUBLICATION_FAILED',result.returncode,flush=True)
 state=subprocess.run(['supervisorctl','status','pi0_va_d0','pi0_va_pipeline'],capture_output=True,text=True)
 statuses=[line.split()[1] for line in state.stdout.splitlines() if len(line.split())>1]
 if (WORK/'complete.json').exists():
  print('EXPERIMENT_COMPLETE_LOGS_SAVED',flush=True);break
 if statuses and all(s in ['EXITED','FATAL','STOPPED'] for s in statuses):
  print('EXPERIMENT_SERVICES_IDLE_OR_FAILED',statuses,flush=True);break
 time.sleep(300)
