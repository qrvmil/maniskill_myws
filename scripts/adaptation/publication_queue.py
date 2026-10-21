"""Managed asynchronous publication of immutable snapshots; no GPU/optimizer changes."""
import argparse,contextlib,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from common import ROOT,WORK,SFT,write
QUEUE=WORK/'publication_queue'

@contextlib.contextmanager
def queue_lock():
 QUEUE.mkdir(exist_ok=True)
 with (QUEUE/'.lock').open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX)
  yield

def version(folder):
 rows=[]
 for p in sorted(Path(folder).rglob('*')):
  if p.is_file() and p.name not in ('inventory.json','upload_receipt.json') and '.cache' not in p.parts:
   s=p.stat();rows.append((str(p.relative_to(folder)),s.st_size,s.st_mtime_ns))
 return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

def submit(folder,remote,full_readback=False):
 with queue_lock():return _submit(folder,remote,full_readback)

def _submit(folder,remote,full_readback=False):
 folder=Path(folder).resolve();QUEUE.mkdir(exist_ok=True)
 digest=version(folder);key=hashlib.sha256((remote+digest).encode()).hexdigest()
 path=QUEUE/(key+'.json')
 if path.exists():
  old=json.loads(path.read_text())
  if old['status']=='done' and (folder/'upload_receipt.json').exists():return json.loads((folder/'upload_receipt.json').read_text())
  return dict(commit=None,queued=True,job=str(path))
 if receipt_matches(dict(folder=str(folder),remote=remote,version=digest)):
  # Reuse a fully verified immutable publication, e.g. the same 100 reset
  # states restored by later methods. No extra commit or repeated readback.
  return json.loads((folder/'upload_receipt.json').read_text())
 priority=0 if remote=='checkpoints/teacher' or remote.endswith('/final') else 2 if remote.startswith('final/') else 5
 parts=remote.split('/')
 if len(parts)==3 and parts[0]=='checkpoints' and parts[1] in SFT and parts[2].isdigit():
  priority=0 if int(parts[2])==SFT[parts[1]]['updates'] else 10
 write(path,dict(folder=str(folder),remote=remote,full_readback=full_readback,version=digest,status='queued',priority=priority,created_at=time.time(),attempts=0))
 _retire_superseded_unlocked()
 print('HF_QUEUED',remote,flush=True)
 return dict(commit=None,queued=True,job=str(path))

def obsolete(job):
 parts=job['remote'].split('/')
 if len(parts)==3 and parts[0]=='checkpoints' and parts[1] in SFT and parts[2].isdigit():
  stage=parts[1]
  if int(parts[2])>=SFT[stage]['updates']:return False
  latest=WORK/'sft'/stage/'latest.json'
  if latest.exists():
   newer=json.loads(latest.read_text());folder=Path(newer['checkpoint'])
   if newer.get('step',0)>int(parts[2]) and (folder/'training_state.pt').exists() and (folder/'model.safetensors').exists():return True
  return (WORK/'evidence'/f'{stage}_complete.json').exists()
 if len(parts)==3 and parts[:2]==['checkpoints','residual'] and parts[2]!='final':
  return (WORK/'evidence/residual_complete.json').exists()
 return False

def retire_superseded():
 with queue_lock():_retire_superseded_unlocked()

def _retire_superseded_unlocked():
 # Queue writers can retire waiting snapshots even while the server is busy on
 # another upload. Never touch a running upload or the latest resumable save.
 for p in QUEUE.glob('*.json'):
  job=json.loads(p.read_text())
  if job['status'] in ('queued','failed') and obsolete(job):discard(job,p)

def discard(job,path):
 job['status']='superseded';write(path,job)
 # These are unselected intermediate snapshots. The final snapshot and teacher have separate protected paths.
 folder=Path(job['folder']);parts=job['remote'].split('/')
 if parts[0]=='checkpoints' and folder.exists():
  latest=(WORK/'sft'/parts[1]/'latest.json') if parts[1] in SFT else WORK/'rl/latest.json'
  current=json.loads(latest.read_text())['checkpoint'] if latest.exists() else None
  if str(folder)!=current:shutil.rmtree(folder)
 print('HF_SUPERSEDED_INTERMEDIATE',job['remote'],flush=True)

def claim_next():
 """Claim one waiting snapshot atomically; never select another active upload."""
 with queue_lock():
  _retire_superseded_unlocked()
  candidates=[]
  for p in QUEUE.glob('*.json'):
   job=json.loads(p.read_text())
   if job['status']=='queued' and job.get('retry_after',0)<=time.time():candidates.append((p,job))
  for path,job in sorted(candidates,key=lambda x:(x[1]['priority'],x[1]['created_at'])):
   if not Path(job['folder']).exists():
    job.update(status='failed',error='snapshot directory missing');write(path,job);continue
   job.update(status='running',attempts=job['attempts']+1);write(path,job)
   return path,job
 return None

def recover_interrupted():
 # Called only by the sole server after its process lock is acquired. All old
 # children died with the supervisor process group; repeating an immutable upload is safe.
 with queue_lock():
  for path in QUEUE.glob('*.json'):
   job=json.loads(path.read_text())
   if job['status']=='running':
    job['status']='queued';write(path,job)
    print('HF_RECOVER_INTERRUPTED',job['remote'],flush=True)

def claim_group():
 """Batch 10–30 episodes, or flush a smaller tail; keep non-episode jobs separate."""
 with queue_lock():
  backoff=QUEUE/'commit-backoff'
  if backoff.exists() and json.loads(backoff.read_text())['until']>time.time():return []
  _retire_superseded_unlocked();candidates=[]
  for path in QUEUE.glob('*.json'):
   job=json.loads(path.read_text())
   if job['status']!='queued' or job.get('retry_after',0)>time.time():continue
   if not Path(job['folder']).exists():
    job.update(status='failed',error='snapshot directory missing');write(path,job);continue
   candidates.append((path,job))
  finals=[x for x in candidates if x[1]['remote'].startswith('final/')]
  flush=(QUEUE/'.flush').exists() or (finals and time.time()-min(j['created_at'] for _,j in finals)>=300)
  if len(finals)<10 and not flush:candidates=[x for x in candidates if not x[1]['remote'].startswith('final/')]
  if not candidates:return []
  first=min(candidates,key=lambda x:(x[1]['priority'],x[1]['created_at']))
  group=sorted(finals,key=lambda x:x[1]['created_at'])[:30] if first[1]['remote'].startswith('final/') else [first]
  for path,job in group:
   job.update(status='running',attempts=job['attempts']+1);write(path,job)
  return group

def receipt_matches(job):
 """A partially failed batch may still have fully verified individual episodes."""
 folder=Path(job['folder']);path=folder/'upload_receipt.json'
 if not path.exists() or version(folder)!=job['version']:return False
 from common import sha
 receipt=json.loads(path.read_text())
 actual={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('inventory.json','upload_receipt.json') and '.cache' not in p.parts}
 expected={v['path'][len(job['remote'])+1:]:v['sha256'] for v in receipt['verified'] if v['path'].startswith(job['remote']+'/')}
 return actual==expected

def serve():
 from storage import push_progress
 QUEUE.mkdir(exist_ok=True);last_push=0;active={}
 server_lock=(QUEUE/'.serve.lock').open('a')
 fcntl.flock(server_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 recover_interrupted()
 while True:
  retire_superseded()
  for key,(group,process) in list(active.items()):
   if process.poll() is None:continue
   del active[key];completed=[]
   for path,job in group:
    if not process.returncode or receipt_matches(job):
     job.update(status='done',completed_at=time.time());completed.append(job)
    else:
     job.update(status='queued' if job['attempts']<5 else 'failed',retry_after=time.time()+60,error='publisher process exit '+str(process.returncode))
     print('HF_PUBLISH_RETRY_OR_ERROR',job['remote'],job['status'],flush=True)
    with queue_lock():write(path,job)
   if completed and (any(j['priority']==0 for j in completed) or time.time()-last_push>=300):
    push_progress('exp: verify queued HF publication '+completed[0]['remote']);last_push=time.time()
  # Three independent folders can be committed/read back concurrently. Each
  # retains its own immutable inventory, commit receipt and full verification.
  while len(active)<3:
   group=claim_group()
   if not group:break
   path,job=group[0]
   env=dict(os.environ,PI0_ARTIFACT_WORKER='1',HF_XET_HIGH_PERFORMANCE='1',CUDA_VISIBLE_DEVICES='')
   env.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
   if job['remote'].startswith('final/'):
    manifest=QUEUE/(path.stem+'.batch');write(manifest,[j for _,j in group])
    cmd=['/venv/main/bin/python','-u',str(ROOT/'scripts/adaptation/publish_episode_batch.py'),str(manifest)]
   else:
    cmd=['/venv/main/bin/python','-u',str(ROOT/'scripts/adaptation/publish_remote.py'),job['folder'],job['remote']]
    if job['full_readback']:cmd.append('--full-readback')
   print('HF_PUBLISH_START',job['remote'],'group_size',len(group),flush=True)
   active[path]=(group,subprocess.Popen(cmd,cwd=ROOT,env=env))
  time.sleep(2 if active else 5)

def drain():
 QUEUE.mkdir(exist_ok=True);(QUEUE/'.flush').touch()
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
