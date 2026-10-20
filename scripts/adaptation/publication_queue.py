"""Managed asynchronous publication of immutable snapshots; no GPU/optimizer changes."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from common import ROOT,WORK,SFT,write
QUEUE=WORK/'publication_queue'

def version(folder):
 rows=[]
 for p in sorted(Path(folder).rglob('*')):
  if p.is_file() and p.name not in ('inventory.json','upload_receipt.json') and '.cache' not in p.parts:
   s=p.stat();rows.append((str(p.relative_to(folder)),s.st_size,s.st_mtime_ns))
 return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

def submit(folder,remote,full_readback=False):
 folder=Path(folder).resolve();QUEUE.mkdir(exist_ok=True)
 digest=version(folder);key=hashlib.sha256((remote+digest).encode()).hexdigest()
 path=QUEUE/(key+'.json')
 if path.exists():
  old=json.loads(path.read_text())
  if old['status']=='done' and (folder/'upload_receipt.json').exists():return json.loads((folder/'upload_receipt.json').read_text())
  return dict(commit=None,queued=True,job=str(path))
 priority=0 if remote=='checkpoints/teacher' or remote.endswith('/final') else 2 if remote.startswith('final/') else 5
 parts=remote.split('/')
 if len(parts)==3 and parts[0]=='checkpoints' and parts[1] in SFT and parts[2].isdigit():
  priority=0 if int(parts[2])==SFT[parts[1]]['updates'] else 10
 write(path,dict(folder=str(folder),remote=remote,full_readback=full_readback,version=digest,status='queued',priority=priority,created_at=time.time(),attempts=0))
 print('HF_QUEUED',remote,flush=True)
 return dict(commit=None,queued=True,job=str(path))

def obsolete(job):
 parts=job['remote'].split('/')
 if len(parts)==3 and parts[0]=='checkpoints' and parts[1] in SFT and parts[2].isdigit():
  stage=parts[1]
  return int(parts[2])<SFT[stage]['updates'] and (WORK/'evidence'/f'{stage}_complete.json').exists()
 if len(parts)==3 and parts[:2]==['checkpoints','residual'] and parts[2]!='final':
  return (WORK/'evidence/residual_complete.json').exists()
 return False

def discard(job,path):
 job['status']='superseded';write(path,job)
 # These are unselected intermediate snapshots. The final snapshot and teacher have separate protected paths.
 folder=Path(job['folder']);parts=job['remote'].split('/')
 if parts[0]=='checkpoints' and folder.exists():
  latest=(WORK/'sft'/parts[1]/'latest.json') if parts[1] in SFT else WORK/'rl/latest.json'
  current=json.loads(latest.read_text())['checkpoint'] if latest.exists() else None
  if str(folder)!=current:shutil.rmtree(folder)
 print('HF_SUPERSEDED_INTERMEDIATE',job['remote'],flush=True)

def serve():
 from storage import push_progress
 QUEUE.mkdir(exist_ok=True);last_push=0
 while True:
  candidates=[]
  for p in QUEUE.glob('*.json'):
   j=json.loads(p.read_text())
   if j['status'] in ('queued','running','failed') and obsolete(j):discard(j,p);continue
   if j['status'] in ('queued','running') and j.get('retry_after',0)<=time.time():candidates.append((p,j))
  if not candidates:time.sleep(5);continue
  path,job=min(candidates,key=lambda x:(x[1]['priority'],x[1]['created_at']))
  if not Path(job['folder']).exists():
   job.update(status='failed',error='snapshot directory missing');write(path,job);continue
  job.update(status='running',attempts=job['attempts']+1);write(path,job)
  env=dict(os.environ,PI0_ARTIFACT_WORKER='1',HF_XET_HIGH_PERFORMANCE='1',CUDA_VISIBLE_DEVICES='')
  env.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
  cmd=['/venv/main/bin/python','-u',str(ROOT/'scripts/adaptation/publish_remote.py'),job['folder'],job['remote']]
  if job['full_readback']:cmd.append('--full-readback')
  print('HF_PUBLISH_START',job['remote'],flush=True)
  process=subprocess.Popen(cmd,cwd=ROOT,env=env)
  while process.poll() is None:
   # Clean obsolete *queued* snapshots while another immutable snapshot is uploading.
   for other in QUEUE.glob('*.json'):
    if other==path:continue
    candidate=json.loads(other.read_text())
    if candidate['status'] in ('queued','failed') and obsolete(candidate):discard(candidate,other)
   time.sleep(5)
  if process.returncode:
   job.update(status='queued' if job['attempts']<5 else 'failed',retry_after=time.time()+60,error='publisher process exit '+str(process.returncode));write(path,job)
   print('HF_PUBLISH_RETRY_OR_ERROR',job['remote'],job['status'],flush=True);continue
  job.update(status='done',completed_at=time.time());write(path,job)
  if job['priority']==0 or time.time()-last_push>=300:
   push_progress('exp: verify queued HF publication '+job['remote']);last_push=time.time()
  # A newly queued final supersedes older unselected snapshots, but an active upload is allowed to finish.

def drain():
 while True:
  jobs=[json.loads(p.read_text()) for p in QUEUE.glob('*.json')] if QUEUE.exists() else []
  failures=[j['remote'] for j in jobs if j['status']=='failed' and not obsolete(j)]
  if failures:raise RuntimeError('Publication failures: '+str(failures))
  pending=[j for j in jobs if j['status'] in ('queued','running') and not obsolete(j)]
  if not pending:return
  print('AWAITING_VERIFIED_HF_PUBLICATIONS',len(pending),[j['remote'] for j in pending[:4]],flush=True)
  time.sleep(30)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['serve','drain']);args=p.parse_args()
 serve() if args.mode=='serve' else drain()
