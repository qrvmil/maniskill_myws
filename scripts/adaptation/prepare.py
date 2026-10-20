"""Audit 100 native demos and compute only D0 normalization with official RunningStats."""
from common import *
def main():
 from openpi.shared import normalize
 audits={}
 for task in TASKS:
  ds=NativeDataset(task,transformed=False);audits[task]=ds.audit
  write(WORK/'evidence'/f'{task}_source_audit.json',ds.audit)
  if task=='D0' and not NORM.exists():
   stats={k:normalize.RunningStats() for k in ('state','actions')}
   # Every aligned frame from all 50 demos, including the incomplete final batch.
   for e in ds.episodes:
    stats['state'].update(e['state'])
    a=e['actions'];chunks=a[np.minimum(np.arange(len(a))[:,None]+np.arange(50)[None,:],len(a)-1)]
    stats['actions'].update(chunks)
   normalize.save(NORM.parent,{k:s.get_statistics() for k,s in stats.items()})
   write(NORM.parent/'provenance.json',dict(training_task='D0',demonstrations=50,aligned_frames=len(ds),
      source_sha256=ds.audit['sha256'],normalization_sha256=sha(NORM),chunk_horizon=50,method='official RunningStats, all aligned frames, repeated-last chunk padding'))
  del ds
 assert json.loads((NORM.parent/'provenance.json').read_text())['normalization_sha256']==sha(NORM)
 write(WORK/'evidence/data_prepared.json',dict(sources=audits,normalization_sha256=sha(NORM)))
 print('ALL_NATIVE_DEMOS_AND_D0_NORM_PREPARED',sha(NORM),flush=True)
if __name__=='__main__':main()
