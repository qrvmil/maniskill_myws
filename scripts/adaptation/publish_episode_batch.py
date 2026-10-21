"""One bounded HF commit for several episodes; full per-episode readback receipts."""
import argparse,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import ROOT,WORK,HF_REPO,sha,write,json

def publish_groups(groups,*,api=None,download=None):
 from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
 api=api or HfApi();download=download or hf_hub_download
 assert 1<=len(groups)<=30
 operations=[];inventories={}
 for job in groups:
  assert job['remote'].startswith('final/')
  folder=Path(job['folder'])
  inventory={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file() and p.name not in ('inventory.json','upload_receipt.json')}
  assert len(inventory)==2 and 'episode.json' in inventory and sum(n.endswith('.mp4') for n in inventory)==1
  write(folder/'inventory.json',inventory);inventories[job['remote']]=inventory
  operations.extend(CommitOperationAdd(path_in_repo=job['remote']+'/'+name,path_or_fileobj=folder/name) for name in (*inventory,'inventory.json'))
 assert len(operations)<=90
 started=time.time()
 commit=api.create_commit(repo_id=HF_REPO,repo_type='model',operations=operations,commit_message=f'Save {len(groups)} final episodes with individual inventories')
 uploaded=time.time();paths=[j['remote']+'/'+n for j in groups for n in inventories[j['remote']]]
 infos={i.path:i for i in api.get_paths_info(HF_REPO,paths,revision=commit.oid)}
 def verify_episode(job):
  verified=[];remote=job['remote']
  for name,meta in inventories[remote].items():
   path=remote+'/'+name;info=infos[path];assert info.size==meta['bytes'],path
   if info.lfs:
    checksum=info.lfs.sha256 if hasattr(info.lfs,'sha256') else info.lfs['sha256'];assert checksum==meta['sha256'],path
   dest=Path(download(HF_REPO,path,revision=commit.oid,local_dir=WORK/'readback'/remote,force_download=True))
   assert sha(dest)==meta['sha256'],path
   verified.append(dict(path=path,verification='download+sha256',sha256=meta['sha256']))
  receipt=dict(repo=HF_REPO,commit=commit.oid,verified=verified,time=time.time(),upload_seconds=uploaded-started,
   verification_seconds=time.time()-uploaded,batch_episodes=len(groups),commit_operations=len(operations))
  write(Path(job['folder'])/'upload_receipt.json',receipt)
  write(ROOT/'experiments/pi0_d0_d1_three_adaptations/evidence'/(remote.replace('/','_')+'.json'),receipt)
  print('HF_VERIFIED',remote,commit.oid,flush=True)
  return receipt
 with ThreadPoolExecutor(max_workers=8) as executor:return list(executor.map(verify_episode,groups))

if __name__=='__main__':
 os.environ['PI0_ARTIFACT_WORKER']='1';os.environ['HF_XET_HIGH_PERFORMANCE']='1'
 os.environ.setdefault('HF_XET_FIXED_UPLOAD_CONCURRENCY','32');os.environ.setdefault('HF_XET_FIXED_DOWNLOAD_CONCURRENCY','16');os.environ.setdefault('HF_XET_CLIENT_READ_TIMEOUT','600s')
 os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
 p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args()
 publish_groups(json.loads(Path(a.manifest).read_text()))
