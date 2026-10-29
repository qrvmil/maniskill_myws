"""A saved candidate must be completed before advancing, even across failures."""
from pathlib import Path
import sys,json
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
import train_sft,evaluate,storage

class Model:
 training=True
 def train(self,mode=True):self.training=mode;return self

def setup(monkeypatch,tmp_path):
 monkeypatch.setattr(train_sft,'WORK',tmp_path)
 checkpoint=tmp_path/'sft/full/250';(checkpoint/'assets/d0').mkdir(parents=True)
 (checkpoint/'model.safetensors').write_bytes(b'candidate250')
 (checkpoint/'training.json').write_text('{"completed_updates":250}')
 (checkpoint/'assets/d0/norm_stats.json').write_text('{"norm_stats":{}}')
 calls=[];publications=[]
 def validate(model,path):
  calls.append(str(path));model.train(False);torch.rand(11);np.random.random(17)
  return dict(checkpoint=str(path),n=10,successes=1,seeds=list(range(22000,22010)),episodes=[dict(seed=22000+i,success=i==0) for i in range(10)])
 monkeypatch.setattr(evaluate,'validation_teacher',validate)
 monkeypatch.setattr(storage,'publish',lambda *a,**k:publications.append(a[1]) or {'queued':True})
 return checkpoint,calls,publications

def test_candidate_reconciliation_restores_rng_and_selects_exact_checkpoint(monkeypatch,tmp_path):
 checkpoint,calls,publications=setup(monkeypatch,tmp_path)
 torch.manual_seed(29);np.random.seed(29);cpu=torch.get_rng_state().clone();nr=np.random.get_state()
 model=Model();train_sft.complete_teacher_candidate(model,checkpoint,250)
 assert torch.equal(cpu,torch.get_rng_state()) and model.training
 after=np.random.get_state();assert nr[0]==after[0] and np.array_equal(nr[1],after[1]) and nr[2:]==after[2:]
 teacher=json.loads((tmp_path/'teacher.json').read_text());assert teacher['step']==250 and teacher['source_checkpoint']==str(checkpoint)
 assert calls==[str(checkpoint)] and publications==['checkpoints/teacher']

def test_crash_after_validation_reuses_complete_evidence(monkeypatch,tmp_path):
 checkpoint,calls,publications=setup(monkeypatch,tmp_path)
 original=train_sft.os.link
 def fail(*a,**k):raise OSError('simulated interrupted link')
 monkeypatch.setattr(train_sft.os,'link',fail)
 with pytest.raises(OSError):train_sft.complete_teacher_candidate(Model(),checkpoint,250)
 assert (tmp_path/'evidence/teacher_validation_250.json').exists()
 monkeypatch.setattr(train_sft.os,'link',original)
 train_sft.complete_teacher_candidate(Model(),checkpoint,250)
 assert len(calls)==1 and (tmp_path/'teacher.json').exists()

def test_crash_after_registration_requeues_teacher_publication(monkeypatch,tmp_path):
 checkpoint,calls,publications=setup(monkeypatch,tmp_path)
 count=[]
 def publish(*a,**k):
  count.append(a[1])
  if len(count)==1:raise OSError('simulated missing enqueue')
  return {'queued':True}
 monkeypatch.setattr(storage,'publish',publish)
 with pytest.raises(OSError):train_sft.complete_teacher_candidate(Model(),checkpoint,250)
 assert (tmp_path/'teacher.json').exists()
 train_sft.complete_teacher_candidate(Model(),checkpoint,250)
 assert len(calls)==1 and count==['checkpoints/teacher','checkpoints/teacher']
