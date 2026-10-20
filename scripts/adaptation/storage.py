"""Publish progress, verify remote bytes, preserve resumable model/optimizer states."""
import json, os, subprocess, time
from pathlib import Path
from common import ROOT,WORK,HF_REPO,sha,write

def publish(folder,remote,*,full_readback=False):
 from huggingface_hub import HfApi,hf_hub_download,hf_hub_url
 from huggingface_hub.utils import get_session
 folder=Path(folder)
 inventory={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('upload_receipt.json','inventory.json')}
 write(folder/'inventory.json',inventory)
 api=HfApi();commit=api.upload_folder(repo_id=HF_REPO,folder_path=folder,path_in_repo=remote,commit_message='Save '+remote)
 # Read back each small artifact; large artifacts require exact remote LFS content hash plus sampled HTTP bytes.
 verified=[]
 for name,meta in inventory.items():
  path=remote+'/'+name
  info=api.get_paths_info(HF_REPO,[path],revision=commit.oid)[0]
  if info.lfs:
   checksum=info.lfs.sha256 if hasattr(info.lfs,'sha256') else info.lfs['sha256']
   assert checksum==meta['sha256'],path
  assert info.size==meta['bytes'],path
  if meta['bytes']<10_000_000 or full_readback:
   # Dedicated location lets us remove large verification copies after hashing.
   tmp=WORK/'readback'/remote
   dest=Path(hf_hub_download(HF_REPO,path,revision=commit.oid,local_dir=tmp,force_download=True))
   assert sha(dest)==meta['sha256'],path
   if meta['bytes']>=10_000_000:dest.unlink()
   method='download+sha256'
  else:
   import requests
   url=hf_hub_url(HF_REPO,path,revision=commit.oid)
   for start in (0,max(0,meta['bytes']-65536)):
    end=min(meta['bytes']-1,start+65535)
    r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=120)
    r.raise_for_status()
    assert r.status_code==206,(path,r.status_code)
    data=r.content
    with open(folder/name,'rb') as f:f.seek(start);assert data==f.read(end-start+1)
   method='remote LFS sha256 + first/last 64KiB readback'
  verified.append(dict(path=path,verification=method,sha256=meta['sha256']))
 receipt=dict(repo=HF_REPO,commit=commit.oid,verified=verified,time=time.time())
 write(folder/'upload_receipt.json',receipt)
 evidence=ROOT/'experiments/pi0_d0_d1_three_adaptations/evidence'/ (remote.replace('/','_')+'.json')
 write(evidence,receipt)
 print('HF_VERIFIED',remote,commit.oid,flush=True)
 return receipt

def push_progress(message):
 dest=ROOT/'experiments/pi0_d0_d1_three_adaptations'
 for src in (WORK/'evidence').glob('*.json'):
  import shutil
  (dest/'evidence').mkdir(exist_ok=True);shutil.copy2(src,dest/'evidence'/src.name)
 subprocess.run(['git','add','scripts/adaptation','experiments/pi0_d0_d1_three_adaptations'],cwd=ROOT,check=True)
 subprocess.run(['git','add','-f',*map(str,(dest/'evidence').glob('*.json'))],cwd=ROOT,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
  subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
  subprocess.run(['git','push','origin','HEAD'],cwd=ROOT,check=True)
