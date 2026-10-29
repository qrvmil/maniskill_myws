"""CPU-only data audit; same native indexing and D0 normalization as baseline."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
BASE=Path('/workspace/visual-alignment-repro-20261029');ROOT=BASE/'maniskill_myws'
sys.path[:0]=[str(ROOT/'scripts/adaptation'),str(ROOT/'src'),str(BASE/'pi0-d0d1-visual-alignment')]
import common
from openpi.shared import normalize
from openpi_client.image_tools import resize_with_pad
from alignment.data import VIEWS,augmentation_params,replay,image_hash
common.ASSETS=BASE/'private';common.NORM=BASE/'private/historical/checkpoints/d0/3000/assets/d0/norm_stats.json'
evidence=BASE/'evidence';private=BASE/'private/smoke';private.mkdir(parents=True,exist_ok=True)
summary={};ds=None
for task in ['D1','D0']:
 ds=common.NativeDataset(task,transformed=False)
 summary[task]={'demonstrations':len(ds.episodes),'aligned_frames':len(ds),'source_sha256':ds.audit['sha256']}
 assert len(ds.episodes)==50
 if task=='D0':
  stats={k:normalize.RunningStats() for k in ('state','actions')}
  for e in ds.episodes:
   stats['state'].update(e['state']);a=e['actions']
   stats['actions'].update(a[np.minimum(np.arange(len(a))[:,None]+np.arange(50)[None,:],len(a)-1)])
  normalize.save(private/'normalization',{k:s.get_statistics() for k,s in stats.items()})
 else:del ds
norm=private/'normalization/norm_stats.json'
assert norm.read_bytes()==common.NORM.read_bytes(),'Normalization differs from historical immutable D0-only bytes'
common.NORM=norm
# Use official original transforms, but deterministic shared source K4 image replay.
from openpi import transforms as T
cfg=common.config();dc=cfg.data.create(cfg.assets_dirs,cfg.model)
transform=T.compose([*dc.data_transforms.inputs,T.Normalize(dc.norm_stats,use_quantiles=False),*dc.model_transforms.inputs])
indices=np.random.default_rng(0).permutation(len(ds))[:8];items=[];identities=[]
for j,index in enumerate(indices):
 raw=ds.raw(int(index));episode,frame=ds.index[index];sample=f'D0/{ds.audit["demo_names"][episode]}/{frame}';aug=j%4
 images=[];hashes=[]
 for key,view in zip(['observation/image','observation/wrist_image'],VIEWS):
  pixels=resize_with_pad(raw[key],224,224)
  pixels=replay(pixels,augmentation_params(sample,aug,view));raw[key]=pixels
  images.append(pixels);hashes.append(image_hash(pixels))
 item=transform(raw)
 for key,pixels in zip(VIEWS,images):
  np.testing.assert_array_equal(item['image'][key],pixels)
 items.append((item,images));identities.append(dict(sample_id=sample,augmentation=aug,image_sha256=hashes))
import pickle
with (private/'batch.pkl').open('wb') as f:pickle.dump(items,f)
summary.update(normalization_sha256=common.sha(norm),historical_norm_exact=True,effective_batch=8,samples=identities,extra_train_augmentation=False)
(evidence/'data-preflight.json').write_text(json.dumps(summary,indent=2)+'\n')
print('DATA_PREFLIGHT_PASSED',json.dumps({k:v for k,v in summary.items() if k!='samples'}),flush=True)
