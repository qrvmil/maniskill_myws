"""Batch commits retain exact paths and fail closed on corrupted readbacks."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/adaptation'))
import publish_episode_batch as batch

def fixture(tmp_path,monkeypatch,corrupt=False):
 work=tmp_path/'work';root=tmp_path/'repo';work.mkdir();root.mkdir()
 monkeypatch.setattr(batch,'WORK',work);monkeypatch.setattr(batch,'ROOT',root)
 groups=[]
 for i in range(2):
  folder=work/str(i);folder.mkdir();(folder/'episode.json').write_text(json.dumps(dict(seed=30000+i)))
  (folder/f'clip{i}.mp4').write_bytes(b'movie'+bytes([i]))
  groups.append(dict(folder=str(folder),remote=f'final/full/D0/{30000+i}'))
 class Api:
  def create_commit(self,**kw):
   self.calls=getattr(self,'calls',0)+1
   self.files={o.path_in_repo:Path(o.path_or_fileobj).read_bytes() for o in kw['operations']}
   return SimpleNamespace(oid='one-batch-commit')
  def get_paths_info(self,repo,paths,revision):
   return [SimpleNamespace(path=p,size=len(self.files[p]),lfs=SimpleNamespace(sha256=hashlib.sha256(self.files[p]).hexdigest())) for p in paths]
 api=Api()
 def download(repo,path,*,revision,local_dir,force_download):
  assert revision=='one-batch-commit' and force_download
  dest=Path(local_dir)/Path(path).name;dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(b'corrupted' if corrupt and path.endswith('clip0.mp4') else api.files[path]);return dest
 return groups,api,download

def test_single_commit_keeps_individual_full_readback_receipts(tmp_path,monkeypatch):
 groups,api,download=fixture(tmp_path,monkeypatch)
 receipts=batch.publish_groups(groups,api=api,download=download)
 assert api.calls==1 and len(api.files)==6 and len(receipts)==2
 for job,r in zip(groups,receipts):
  assert r['commit']=='one-batch-commit' and len(r['verified'])==2
  assert all(v['path'].startswith(job['remote']+'/') and v['verification']=='download+sha256' for v in r['verified'])
  assert json.loads((Path(job['folder'])/'upload_receipt.json').read_text())==r

def test_corrupted_video_never_gets_a_success_receipt(tmp_path,monkeypatch):
 groups,api,download=fixture(tmp_path,monkeypatch,corrupt=True)
 with pytest.raises(AssertionError):batch.publish_groups(groups,api=api,download=download)
 assert not (Path(groups[0]['folder'])/'upload_receipt.json').exists()
