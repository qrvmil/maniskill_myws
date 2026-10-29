"""Committed residual snapshots survive interrupted latest/enqueue writes."""
from pathlib import Path
import sys,json,hashlib
import pytest,torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
import train_residual,storage,publication_queue

def checkpoint(root,label,active,environment,episode,teacher):
 p=root/label;(p/'assets/d0').mkdir(parents=True)
 (p/'residual.pt').write_bytes(('model'+label).encode());(p/'online_replay.npz').write_bytes(b'replay')
 (p/'assets/d0/norm_stats.json').write_bytes(b'norm')
 counters=dict(active_steps=active,environment_steps=environment,episode=episode)
 torch.save(dict(counters=counters,calql_done=True,rows=[]),p/'training_state.pt')
 if environment:(p/'collection_traces.tar.gz').write_bytes(b'traces')
 sha=lambda f:hashlib.sha256((p/f).read_bytes()).hexdigest()
 data=dict(counters=counters,base_weights_sha256='base',offline_replay_sha256='offline',normalization_sha256=sha('assets/d0/norm_stats.json'),teacher=teacher,
  gamma=.99,beta=.1,residual_scale=2.,frozen_d0_weights_unchanged=True,
  weights_sha256=sha('residual.pt'),training_state_sha256=sha('training_state.pt'),online_replay_sha256=sha('online_replay.npz'))
 (p/'training.json').write_text(json.dumps(data));return p

def setup(monkeypatch,tmp_path):
 monkeypatch.setenv('PI0_VISUAL_ALIGNMENT','1');norm=tmp_path/'norm.json';norm.write_bytes(b'norm');monkeypatch.setattr(train_residual,'NORM',norm)
 teacher=dict(step=250,validation=dict(weights_sha256='teacher',successes=1),canonical_validation=dict(successes=1))
 events=[]
 def publish(folder,remote,**kwargs):
  events.append(('publish',remote));(folder/'upload_receipt.json').write_text('{"commit":"test"}');return {}
 monkeypatch.setattr(storage,'publish',publish);monkeypatch.setattr(publication_queue,'drain',lambda:events.append(('drain',)))
 root=tmp_path/'rl';root.mkdir();return root,teacher,events

@pytest.mark.parametrize('label,active,environment,episode',[('after_calql',0,0,0),('after_teacher_warmup',0,4127,20),('active_5060',5060,9187,43),('final',50000,54127,248)])
def test_unindexed_residual_checkpoint_repairs_pointer_and_is_durable(monkeypatch,tmp_path,label,active,environment,episode):
 root,teacher,events=setup(monkeypatch,tmp_path);target=checkpoint(root,label,active,environment,episode,teacher)
 result=train_residual.reconcile_residual_checkpoint(root,'base','offline',teacher)
 assert result['checkpoint']==str(target) and result['counters']['active_steps']==active
 assert json.loads((root/'latest.json').read_text())==result
 assert events==[('publish','checkpoints/residual/'+label),('drain',)]

def test_final_wins_counter_tie_and_partials_are_ignored(monkeypatch,tmp_path):
 root,teacher,events=setup(monkeypatch,tmp_path)
 checkpoint(root,'active_50000',50000,54127,248,teacher);final=checkpoint(root,'final',50000,54127,248,teacher)
 (root/'future.partial').mkdir();(root/'future.partial/training.json').write_text('{}')
 result=train_residual.reconcile_residual_checkpoint(root,'base','offline',teacher)
 assert result['checkpoint']==str(final)

def test_highest_incomplete_commit_is_not_silently_ignored(monkeypatch,tmp_path):
 root,teacher,events=setup(monkeypatch,tmp_path)
 checkpoint(root,'after_calql',0,0,0,teacher);bad=checkpoint(root,'active_5060',5060,9187,43,teacher);(bad/'training_state.pt').unlink()
 with pytest.raises((ValueError,AssertionError)):train_residual.reconcile_residual_checkpoint(root,'base','offline',teacher)

def test_corrupt_or_mismatched_state_counters_close_gate(monkeypatch,tmp_path):
 root,teacher,events=setup(monkeypatch,tmp_path);p=checkpoint(root,'active_5060',5060,9187,43,teacher)
 state=dict(counters=dict(active_steps=5061,environment_steps=9188,episode=43),calql_done=True);torch.save(state,p/'training_state.pt')
 # Even when file digest is valid, semantic counters must match the metadata.
 data=json.loads((p/'training.json').read_text());data['training_state_sha256']=hashlib.sha256((p/'training_state.pt').read_bytes()).hexdigest();(p/'training.json').write_text(json.dumps(data))
 with pytest.raises((ValueError,AssertionError)):train_residual.reconcile_residual_checkpoint(root,'base','offline',teacher)
