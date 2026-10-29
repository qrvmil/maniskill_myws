"""Final artifact audit with actual counters and no historical hardcoded outcomes."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts/adaptation'))
from common import *

def main():
 from huggingface_hub import HfApi
 from report import analyze,load_rows
 rows=load_rows();cells,pairs=analyze(rows)
 assert cells==json.loads((EXPERIMENT/'results/success_rates.json').read_text())
 selection=json.loads((WORK/'final_selection.json').read_text())
 folders=[WORK/'sft/d0/3000',WORK/'teacher_policy']+[Path(selection[m]['checkpoint']) for m in ['full','expert','residual']]
 expected={'README.md','REPORT.md','VIDEOS.md','PREREGISTRATION.md','SOURCE_AUDIT.md','normalization/d0/norm_stats.json','offline/replay.npz','selection/final_selection.json'}
 policies=[]
 for folder in folders:
  weight='residual.pt' if (folder/'residual.pt').exists() else 'model.safetensors'
  assert sha(folder/'assets/d0/norm_stats.json')==sha(NORM)
  receipt=json.loads((folder/'upload_receipt.json').read_text())
  record=next(r for r in receipt['verified'] if r['path'].endswith('/'+weight))
  assert record['verification']=='download+sha256' and record['sha256']==sha(folder/weight)
  expected.update(r['path'] for r in receipt['verified']);expected.add(record['path'].rsplit('/',1)[0]+'/inventory.json')
  policies.append(dict(path=str(folder),weights_sha256=record['sha256'],hf_commit=receipt['commit']))
 for method in ['full','expert']:
  metadata=json.loads((WORK/'sft'/method/'1500/training.json').read_text())
  assert metadata['completed_updates']==1500 and metadata['initial_weights_sha256']==sha(WORK/'sft/d0/3000/model.safetensors')
  assert metadata['frozen_scope_verified']
 np.testing.assert_array_equal(np.load(WORK/'sft/full/1500/sample_indices.npy'),np.load(WORK/'sft/expert/1500/sample_indices.npy'))
 audit=json.loads((WORK/'evidence/online_audit_final.json').read_text());assert audit['passed'] and audit['active_transitions']==50000
 counters=json.loads((WORK/'rl/final/training.json').read_text())['counters']
 assert audit['transitions']==counters['environment_steps']==50000+audit['warmup_transitions']
 for row in rows:
  folder=WORK/'final'/row['method']/row['task']/str(row['seed'])
  receipt=json.loads((folder/'upload_receipt.json').read_text())
  video=next(r for r in receipt['verified'] if r['path'].endswith('/'+row['video']))
  assert video['verification']=='download+sha256' and video['sha256']==row['video_sha256']
  remote=str(folder.relative_to(WORK));expected.update([remote+'/episode.json',remote+'/'+row['video'],remote+'/inventory.json'])
 resets=list((WORK/'reset_states/final').glob('*/*.npz'));assert len(resets)==100
 expected.update(str(p.relative_to(WORK)) for p in resets)
 expected.update('results/'+name for name in ['episodes.json','episodes.csv','success_rates.json','success_rates.csv','paired_comparisons.json','completeness.json'])
 api=HfApi();head=api.model_info(HF_REPO).sha;files=set(api.list_repo_files(HF_REPO,revision=head))
 assert expected<=files,sorted(expected-files)
 assert sum(p.startswith('final/') and p.endswith('.mp4') for p in files)==300
 assert sum(p.startswith('final/') and p.endswith('/episode.json') for p in files)==300
 result=dict(passed=True,episodes=300,videos=300,reset_states=100,policies=policies,actual_environment_steps=counters['environment_steps'],active_transitions=50000,warmup_transitions=audit['warmup_transitions'],paired_SFT_sampler_exact=True,remote_head=head,required_remote_paths=len(expected))
 write(WORK/'evidence/alignment_delivery_audit.json',result)
 from storage import push_progress
 push_progress('exp: verify all new policies,300remote episodes/videos and actual replay counters')
 print('ALIGNED_EXPERIMENT_DELIVERY_VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
