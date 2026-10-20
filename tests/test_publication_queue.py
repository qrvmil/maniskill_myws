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
