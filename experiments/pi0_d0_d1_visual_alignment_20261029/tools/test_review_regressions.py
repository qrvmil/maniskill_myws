"""Regression cases found by the independent runtime review; CPU only."""
from pathlib import Path
import sys,json,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
import train_sft,publication_queue,pipeline

def test_profile_reaches_data_loading_without_local_os_shadow(monkeypatch):
 class ReachedDataset(Exception):pass
 def dataset(*args,**kwargs):raise ReachedDataset
 monkeypatch.setattr(train_sft,'NativeDataset',dataset)
 with pytest.raises(ReachedDataset):train_sft.train('d0',profile=True)

def test_teacher_is_selected_only_at_registered_early_candidates():
 selected=[step for step in range(1501) if train_sft.is_teacher_candidate('full',step,False)]
 assert selected==[250,500,750,1000,1250]
 assert not any(train_sft.is_teacher_candidate('expert',step,False) for step in range(1501))
 assert not any(train_sft.is_teacher_candidate('full',step,True) for step in range(1501))

def test_new_local_save_cannot_delete_unpublished_snapshot(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');monkeypatch.setattr(publication_queue,'WORK',tmp_path);monkeypatch.setattr(publication_queue,'QUEUE',tmp_path/'publication_queue')
 old=tmp_path/'sft/d0/1000';new=tmp_path/'sft/d0/2000'
 for folder in [old,new]:
  folder.mkdir(parents=True);(folder/'model.safetensors').write_bytes(b'policy');(folder/'training_state.pt').write_bytes(b'optimizer')
 (new.parent/'latest.json').write_text(json.dumps({'checkpoint':str(new),'step':2000}))
 publication_queue.QUEUE.mkdir()
 path=publication_queue.QUEUE/'unpublished.json'
 job={'folder':str(old),'remote':'checkpoints/d0/1000','status':'queued'};path.write_text(json.dumps(job))
 publication_queue.retire_superseded()
 assert old.exists() and (old/'training_state.pt').exists()
 assert json.loads(path.read_text())['status']=='queued'
 assert not publication_queue.obsolete(job)

def test_offline_replay_is_durable_before_residual_updates(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');monkeypatch.setattr(pipeline,'WORK',tmp_path)
 assets=tmp_path/'assets';assets.mkdir();(assets/'serl_resnet10.pt').write_bytes(b'exists');monkeypatch.setattr(pipeline,'ASSETS',assets)
 (tmp_path/'evidence').mkdir();(tmp_path/'evidence/d0_complete.json').write_text('{}');(tmp_path/'teacher.json').write_text('{}')
 events=[]
 class StoppedBeforeTraining(Exception):pass
 def call(script,*args):
  events.append((script,args))
  if script=='train_residual.py' and not args:raise StoppedBeforeTraining
 monkeypatch.setattr(pipeline,'call',call);monkeypatch.setattr(pipeline,'barrier',lambda:events.append(('BARRIER',())))
 monkeypatch.setattr(pipeline,'service_status',lambda name:'EXITED')
 monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,0))
 with pytest.raises(StoppedBeforeTraining):pipeline.main()
 i=events.index(('offline_replay.py',()))
 assert ('BARRIER',()) in events[i+1:-1]
