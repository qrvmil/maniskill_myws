"""Independent delivery check and actual notebook rerun without new episodes."""
from common import *

def final_manifest():
 return {str(p.relative_to(WORK)):sha(p) for root in ('final','reset_states/final')
         for p in sorted((WORK/root).rglob('*')) if p.is_file() and p.suffix in ('.json','.mp4','.npz')}

def main():
 import nbformat
 from nbclient import NotebookClient
 from storage import push_progress
 experiment=ROOT/'experiments/pi0_d0_d1_three_adaptations'
 complete=json.loads((WORK/'complete.json').read_text())
 assert complete['objective_complete'] and complete['episodes']==complete['videos']==300
 selection=json.loads((WORK/'final_selection.json').read_text())
 required=[WORK/'sft/d0/3000',WORK/'teacher_policy']+[Path(selection[m]['checkpoint']) for m in ('full','expert','residual')]
 policies=[]
 for folder in required:
  assert sha(folder/'assets/d0/norm_stats.json')==sha(NORM)
  receipt=json.loads((folder/'upload_receipt.json').read_text())
  weights='residual.pt' if (folder/'residual.pt').exists() else 'model.safetensors'
  verified=next(v for v in receipt['verified'] if v['path'].endswith('/'+weights))
  assert verified['verification']=='download+sha256' and verified['sha256']==sha(folder/weights)
  policies.append(dict(path=str(folder),weights_sha256=verified['sha256'],hf_commit=receipt['commit']))
 episodes=list((WORK/'final').glob('*/*/*/episode.json'));assert len(episodes)==300
 expected_remote=set()
 for path in episodes:
  row=json.loads(path.read_text());receipt=json.loads((path.parent/'upload_receipt.json').read_text())
  video=next(v for v in receipt['verified'] if v['path'].endswith('/'+row['video']))
  assert video['verification']=='download+sha256' and video['sha256']==row['video_sha256']
  remote=str(path.parent.relative_to(WORK));expected_remote.update((remote+'/episode.json',remote+'/'+row['video'],remote+'/inventory.json'))
 assert len(list((WORK/'reset_states/final').glob('*/*.npz')))==100
 expected_remote.update(str(p.relative_to(WORK)) for p in (WORK/'reset_states/final').glob('*/*.npz'))
 for folder in required:
  receipt=json.loads((folder/'upload_receipt.json').read_text())
  expected_remote.update(v['path'] for v in receipt['verified'])
  weights=next(v['path'] for v in receipt['verified'] if v['path'].endswith(('model.safetensors','residual.pt')))
  expected_remote.add(weights.rsplit('/',1)[0]+'/inventory.json')
 residual_receipt=json.loads((WORK/'rl/final/upload_receipt.json').read_text())
 replay=next(v for v in residual_receipt['verified'] if v['path'].endswith('/online_replay.npz'))
 assert replay['verification']=='download+sha256' and replay['sha256']==sha(WORK/'rl/final/online_replay.npz')
 expected_remote.update(('README.md','seminar/seminar.ipynb','seminar/REPORT.md','seminar/VIDEOS.md',
  'seminar/PREREGISTRATION.md','normalization/d0/norm_stats.json','selection/final_selection.json','offline/replay.npz',
  'results/episodes.json','results/episodes.csv','results/success_rates.json','results/paired_comparisons.json','results/completeness.json'))
 # Receipts verify immutable historical commits. Also verify that every requested
 # artifact still exists together at the current head after concurrent uploads.
 from huggingface_hub import HfApi
 api=HfApi();remote_head=api.repo_info(HF_REPO).sha;remote_files=set(api.list_repo_files(HF_REPO,revision=remote_head))
 assert expected_remote<=remote_files,sorted(expected_remote-remote_files)
 assert sum(p.startswith('final/') and p.endswith('.mp4') for p in remote_files)==300
 assert sum(p.startswith('final/') and p.endswith('/episode.json') for p in remote_files)==300
 audit=json.loads((WORK/'evidence/online_audit_final.json').read_text())
 assert audit['passed'] and audit['active_transitions']==50000 and audit['transitions']==54127
 notebook=experiment/'seminar.ipynb';nb=nbformat.read(notebook,as_version=4)
 def check_notebook(nb):
  code=[c for c in nb.cells if c.cell_type=='code']
  assert all(c.execution_count is not None for c in code)
  outputs=[o for c in code for o in c.outputs]
  assert not any(o.output_type=='error' for o in outputs)
  assert sum('data:video/mp4;base64,' in str(o.get('data',{}).get('text/html','')) for o in outputs)==3
  actions=[c for c in code if 'action = policy.action()' in c.source];assert len(actions)==3
  assert all('checked_against_recorded_first_action' in str(c.outputs) and 'True' in str(c.outputs) for c in actions)
  return len(code)
 cells=check_notebook(nb);before=final_manifest()
 NotebookClient(nb,timeout=1800,kernel_name='pi0_three',resources={'metadata':{'path':str(ROOT)}}).execute()
 assert check_notebook(nb)==cells and before==final_manifest(),'Notebook rerun changed final observations or artifacts'
 out=WORK/'notebook_rerun';out.mkdir(exist_ok=True);nbformat.write(nb,out/'seminar.ipynb')
 proof=dict(passed=True,required_policies=policies,final_episodes=300,verified_remote_videos=300,
  unique_task_reset_states=100,executed_notebook_code_cells=cells,embedded_videos=3,
  notebook_rerun_executed=True,final_manifest_unchanged=True,manifest_files=len(before),
  residual_final_replay_audit_passed=True,remote_head=remote_head,
  remote_head_required_paths_verified=len(expected_remote))
 write(WORK/'evidence/delivery_audit.json',proof)
 push_progress('exp: verify five published policies, 300 remote videos and idempotent executed notebook rerun')
 print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
