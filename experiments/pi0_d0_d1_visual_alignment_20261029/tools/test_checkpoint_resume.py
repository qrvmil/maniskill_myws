"""A renamed checkpoint is authoritative even if latest/enqueue was interrupted."""
from pathlib import Path
import json,sys,hashlib
import pytest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
import train_sft,storage

def checkpoint(root,step,initial='initial'):
 p=root/str(step);(p/'assets/d0').mkdir(parents=True)
 weights=b'weights'+str(step).encode();norm=b'fixednorm'
 (p/'model.safetensors').write_bytes(weights);(p/'training_state.pt').write_bytes(b'optimizer_rng')
 (p/'assets/d0/norm_stats.json').write_bytes(norm);(p/'visual_alignment.json').write_text('{}')
 (p/'training.json').write_text(json.dumps(dict(stage='d0',completed_updates=step,initial_weights_sha256=initial,weights_sha256=hashlib.sha256(weights).hexdigest(),normalization_sha256=hashlib.sha256(norm).hexdigest())))
 return p

def test_latest_pointer_and_enqueue_recover_after_rename(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');root=tmp_path/'d0';root.mkdir()
 old=checkpoint(root,1);new=checkpoint(root,1000)
 (root/'latest.json').write_text(json.dumps(dict(checkpoint=str(old),step=1)))
 published=[];monkeypatch.setattr(storage,'publish',lambda folder,remote,**kwargs:published.append((folder,remote)) or {})
 folder,step=train_sft.reconcile_checkpoint(root,'d0',tmp_path/'base','initial')
 assert folder==new and step==1000
 assert json.loads((root/'latest.json').read_text())==dict(checkpoint=str(new),step=1000)
 assert published==[(new,'checkpoints/d0/1000')]

def test_incomplete_partial_is_not_a_committed_checkpoint(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');root=tmp_path/'d0';root.mkdir();old=checkpoint(root,1)
 (root/'1000.partial').mkdir();(root/'1000.partial/model.safetensors').write_bytes(b'unfinished')
 monkeypatch.setattr(storage,'publish',lambda *a,**k:{})
 folder,step=train_sft.reconcile_checkpoint(root,'d0',tmp_path/'base','initial')
 assert folder==old and step==1

def test_changed_weights_in_committed_checkpoint_close_gate(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');root=tmp_path/'d0';root.mkdir();p=checkpoint(root,1)
 (p/'model.safetensors').write_bytes(b'corrupt')
 with pytest.raises((ValueError,AssertionError)):train_sft.reconcile_checkpoint(root,'d0',tmp_path/'base','initial')

def test_resumed_first_update_must_be_remote_before_model_loading(monkeypatch,tmp_path):
 import publication_queue
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');monkeypatch.setattr(train_sft,'WORK',tmp_path)
 (tmp_path/'evidence').mkdir();(tmp_path/'evidence/runtime_review.json').write_text('{"passed":true}')
 assets=tmp_path/'assets';base=assets/'pi0_base_torch';base.mkdir(parents=True);(base/'model.safetensors').write_bytes(b'initialweights');monkeypatch.setattr(train_sft,'ASSETS',assets)
 root=tmp_path/'sft/d0';root.mkdir(parents=True);checkpoint(root,1,hashlib.sha256(b'initialweights').hexdigest())
 monkeypatch.setattr(train_sft,'NativeDataset',lambda *a,**k:object())
 monkeypatch.setattr(storage,'publish',lambda *a,**k:{'queued':True})
 class ReachedBarrier(Exception):pass
 class LoadedTooSoon(Exception):pass
 def drain():raise ReachedBarrier
 def model(*a,**k):raise LoadedTooSoon
 monkeypatch.setattr(publication_queue,'drain',drain);monkeypatch.setattr(train_sft,'load_model',model)
 with pytest.raises(ReachedBarrier):train_sft.train('d0')
