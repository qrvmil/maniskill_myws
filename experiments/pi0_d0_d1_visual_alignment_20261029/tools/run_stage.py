"""Serial, fail-closed controller for the approved aligned common D0 base."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
from common import WORK,EXPERIMENT,HF_REPO,sha,write
from pipeline import service_status
TOOLS=EXPERIMENT/'tools';BASE=WORK.parent

def call(path,*args):
 print('ALIGNED_D0_STAGE',path,args,flush=True)
 subprocess.run([sys.executable,'-u',str(path),*args],cwd=ROOT,check=True)

def snapshot(message):
 subprocess.run(['/venv/main/bin/python','-u',str(TOOLS/'save_progress.py'),message],cwd=ROOT,check=True)

def ready():
 assert json.loads((BASE/'evidence/vram-gate.json').read_text())['passed']
 assert json.loads((BASE/'evidence/model-smoke.json').read_text())['passed']
 assert (BASE/'evidence/model-preflight.exitcode').read_text().strip()=='0'
 assert json.loads((WORK/'evidence/simulator_smoke.json').read_text())['passed']
 assert (WORK/'preflight_media/upload_receipt.json').exists()
 if shutil.disk_usage(WORK).free<64*(1<<30):raise RuntimeError('Need64GiB free disk for full checkpoint+readback and margin')
 from huggingface_hub import HfApi
 api=HfApi();assert api.whoami()['name']=='qrvmil' and not api.model_info(HF_REPO).private

def main():
 ready()
 while service_status('pi0_va_cache_d0') in ('STARTING','RUNNING','STOPPING'):
  print('WAITING_FOR_COMPLETE_VERIFIED_D0_CACHE',flush=True);time.sleep(10)
 assert (BASE/'evidence/alignment-cache-D0.exitcode').read_text().strip()=='0'
 assert json.loads((WORK/'alignment/cache_D0.json').read_text())['teacher_unchanged']
 assert (WORK/'alignment/published_cache_metadata/D0/upload_receipt.json').exists()
 if (WORK/'evidence/d0_complete.json').exists():
  path=WORK/'sft/d0/3000';meta=json.loads((path/'training.json').read_text())
  assert sha(path/'model.safetensors')==meta['weights_sha256']
  assert (path/'upload_receipt.json').exists();print('ALIGNED_D0_ALREADY_COMPLETE',flush=True);return
 snapshot('exp: save verified cache and preflight before D0 projector calibration')
 call(ROOT/'scripts/adaptation/prepare.py')
 call(TOOLS/'prepare_alignment.py','calibrate')
 snapshot('exp: save frozen D0 projector before lambda pilot')
 call(TOOLS/'prepare_alignment.py','pilot')
 snapshot('exp: save D0 lambda decision before integrated training profile')
 profile=WORK/'evidence/memory_profile.json'
 if not profile.exists():call(ROOT/'scripts/adaptation/train_sft.py','d0','--profile')
 p=json.loads(profile.read_text());assert p['effective_batch']==8 and p['microbatch']==2 and p['fp32_master']
 snapshot('exp: save integrated actual-cache training profile before long D0 SFT')
 call(ROOT/'scripts/adaptation/train_sft.py','d0')
 call(ROOT/'scripts/adaptation/publication_queue.py','drain')
 snapshot('exp: verify complete aligned D0 checkpoint before D1 adaptations')

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(WORK/'evidence/d0_controller_failure.json',dict(type=type(exc).__name__,message=str(exc),utc=time.time(),long_stages_must_not_continue=True))
  raise
