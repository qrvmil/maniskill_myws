"""Protect immutable snapshots, deduplicate retries, and retain the selected teacher."""
import json,os,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/adaptation'))
import publication_queue as queue

@pytest.fixture
def isolated(tmp_path,monkeypatch):
 work=tmp_path/'work';work.mkdir()
 monkeypatch.setattr(queue,'WORK',work);monkeypatch.setattr(queue,'QUEUE',work/'queue')
 return work

def test_retry_deduplicates_immutable_snapshot(isolated):
 folder=isolated/'snapshot';folder.mkdir();(folder/'weights.bin').write_bytes(b'weights')
 first=queue.submit(folder,'checkpoints/d0/1000')
 (folder/'inventory.json').write_text('{}')
 second=queue.submit(folder,'checkpoints/d0/1000')
 assert first['job']==second['job'] and len(list(queue.QUEUE.glob('*.json')))==1
 assert json.loads(Path(first['job']).read_text())['status']=='queued'

def test_final_and_teacher_never_become_obsolete(isolated):
 (isolated/'evidence').mkdir();(isolated/'evidence/d0_complete.json').write_text('{}')
 assert queue.obsolete(dict(remote='checkpoints/d0/1000'))
 assert not queue.obsolete(dict(remote='checkpoints/d0/3000'))
 assert not queue.obsolete(dict(remote='checkpoints/teacher'))

def test_teacher_bytes_survive_unselected_intermediate_cleanup(isolated):
 old=isolated/'sft/full/250';old.mkdir(parents=True);(old/'model.safetensors').write_bytes(b'exact teacher bytes')
 teacher=isolated/'teacher_policy';teacher.mkdir();os.link(old/'model.safetensors',teacher/'model.safetensors')
 (isolated/'sft/full/latest.json').write_text(json.dumps(dict(checkpoint=str(isolated/'sft/full/1500'))))
 job=dict(remote='checkpoints/full/250',folder=str(old),status='queued')
 marker=isolated/'job.json';queue.discard(job,marker)
 assert not old.exists() and (teacher/'model.safetensors').read_bytes()==b'exact teacher bytes'
 assert json.loads(marker.read_text())['status']=='superseded'

def test_waiting_snapshot_retired_only_after_newer_resumable_save(isolated):
 old=isolated/'sft/full/250';old.mkdir(parents=True);(old/'model.safetensors').write_bytes(b'old')
 first=queue.submit(old,'checkpoints/full/250')
 newer=isolated/'sft/full/500';newer.mkdir();(newer/'model.safetensors').write_bytes(b'new')
 (newer.parent/'latest.json').write_text(json.dumps(dict(checkpoint=str(newer),step=500)))
 queue.retire_superseded()
 assert old.exists(), 'Weights alone are insufficient to resume optimization'
 (newer/'training_state.pt').write_bytes(b'optimizer and RNG')
 queue.retire_superseded()
 assert not old.exists() and newer.exists()
 assert json.loads(Path(first['job']).read_text())['status']=='superseded'

def test_queue_writer_preserves_running_upload(isolated):
 old=isolated/'sft/full/250';old.mkdir(parents=True);(old/'model.safetensors').write_bytes(b'upload source')
 first=queue.submit(old,'checkpoints/full/250');path=Path(first['job'])
 job=json.loads(path.read_text());job['status']='running';path.write_text(json.dumps(job))
 (isolated/'evidence').mkdir();(isolated/'evidence/full_complete.json').write_text('{}')
 queue.retire_superseded()
 assert old.exists() and json.loads(path.read_text())['status']=='running'

def test_parallel_claims_never_duplicate_active_snapshot(isolated):
 from concurrent.futures import ThreadPoolExecutor
 for i in range(5):
  folder=isolated/str(i);folder.mkdir();(folder/'episode.json').write_text('{}')
  queue.submit(folder,f'final/full/D0/{30000+i}')
 with ThreadPoolExecutor(max_workers=8) as pool:claimed=list(pool.map(lambda _:queue.claim_next(),range(8)))
 paths=[path for result in claimed if result is not None for path,job in [result]]
 assert len(paths)==len(set(paths))==5
 assert queue.claim_next() is None
 assert all(json.loads(p.read_text())['attempts']==1 for p in paths)

def test_interrupted_upload_recovered_without_changing_snapshot(isolated):
 folder=isolated/'episode';folder.mkdir();(folder/'episode.json').write_text('{"seed":30000}')
 queue.submit(folder,'final/full/D0/30000');path,job=queue.claim_next()
 digest=queue.version(folder);queue.recover_interrupted()
 recovered_path,recovered=queue.claim_next()
 assert recovered_path==path and recovered['attempts']==2
 assert recovered['version']==job['version']==digest==queue.version(folder)
 assert (folder/'episode.json').read_text()=='{"seed":30000}'

def test_batch_respects_backoff_and_operation_bound(isolated):
 import time
 for i in range(35):
  folder=isolated/str(i);folder.mkdir();(folder/'episode.json').write_text('{}')
  queue.submit(folder,f'final/full/D0/{30000+i}')
 (queue.QUEUE/'commit-backoff').write_text(json.dumps(dict(until=time.time()+60)))
 assert queue.claim_group()==[]
 (queue.QUEUE/'commit-backoff').unlink()
 first=queue.claim_group();assert len(first)==30 and len({p for p,j in first})==30
 assert queue.claim_group()==[], 'Small tail waits for the flush barrier or age threshold'
 (queue.QUEUE/'.flush').touch();second=queue.claim_group();assert len(second)==5
 assert not {p for p,j in first}&{p for p,j in second}

def test_previously_verified_identical_reset_tree_reused(isolated):
 from hashlib import sha256
 folder=isolated/'resets';folder.mkdir();payload=b'fixed reset bytes';(folder/'state.npz').write_bytes(payload)
 receipt=dict(commit='already-verified',verified=[dict(path='reset_states/state.npz',sha256=sha256(payload).hexdigest(),verification='download+sha256')])
 (folder/'upload_receipt.json').write_text(json.dumps(receipt))
 assert queue.submit(folder,'reset_states')==receipt
 assert not list(queue.QUEUE.glob('*.json'))
 (folder/'state.npz').write_bytes(b'new reset')
 assert queue.submit(folder,'reset_states')['queued']
