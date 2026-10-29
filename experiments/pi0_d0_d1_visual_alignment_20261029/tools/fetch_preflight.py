"""Fetch only immutable inputs for a discarded smoke, never historical optimizers."""
from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
BASE=Path('/workspace/visual-alignment-repro-20261029')
PRIVATE=BASE/'private'
EVIDENCE=BASE/'evidence'
REPO='qrvmil/pi0-d0-d1-three-adaptations-20261020'
REV='57b978774e8e177d8d721b71f15542751e0610d9'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
api=HfApi();receipts=[]
for stage in ['d0/3000','full/1500','expert/1500','teacher','residual/final']:
 name=f'checkpoints/{stage}/training.json'
 p=hf_hub_download(REPO,name,revision=REV,local_dir=PRIVATE/'historical')
 target=EVIDENCE/('historical_'+stage.replace('/','_')+'_training.json')
 target.write_bytes(Path(p).read_bytes())
 print('HISTORICAL_METADATA',stage,flush=True)
for name in ['checkpoints/d0/3000/model.safetensors','checkpoints/d0/3000/assets/d0/norm_stats.json']:
 p=hf_hub_download(REPO,name,revision=REV,local_dir=PRIVATE/'historical')
 info=api.get_paths_info(REPO,[name],revision=REV)[0]
 digest=sha(p)
 if info.lfs: assert digest==info.lfs.sha256
 receipts.append(dict(repo=REPO,revision=REV,path=name,bytes=Path(p).stat().st_size,sha256=digest))
 print('SMOKE_INPUT_VERIFIED',name,digest,flush=True)
for task in ['pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate','pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate']:
 repo='yifengzhu-hf/LIBERO-datasets';rev='f13aa24a3da8c43c7225569f28c562979fa0e35a';name='libero_spatial/'+task+'_demo.hdf5'
 p=hf_hub_download(repo,name,repo_type='dataset',revision=rev,local_dir=PRIVATE/'native')
 digest=sha(p); info=api.get_paths_info(repo,[name],repo_type='dataset',revision=rev)[0]
 assert digest==info.lfs.sha256
 receipts.append(dict(repo=repo,revision=rev,path=name,bytes=Path(p).stat().st_size,sha256=digest))
 print('NATIVE_VERIFIED',name,digest,flush=True)
repo='nvidia/C-RADIOv3-L';rev='9d0413465e8a91e67bbf2c1ad342815478d1b906'
snapshot_download(repo,revision=rev,local_dir=PRIVATE/'teacher',allow_patterns=['*.py','config.json','model.safetensors','README.md'])
for p in sorted((PRIVATE/'teacher').glob('*')):
 if p.is_file():receipts.append(dict(repo=repo,revision=rev,path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
(EVIDENCE/'smoke-input-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print('PREFLIGHT_INPUTS_READY',flush=True)
