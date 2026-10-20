"""Pinned native D0/D1 and official pi0_base download with remote checksums."""
import base64, concurrent.futures, hashlib, json, os
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, hf_hub_download
ROOT=Path('/workspace/pi0_three/assets')
REV='f13aa24a3da8c43c7225569f28c562979fa0e35a'
TASKS={'D0':'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate','D1':'pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate'}
def digest(path,kind):
 h=hashlib.new(kind)
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h

def native(task):
 name='libero_spatial/'+TASKS[task]+'_demo.hdf5'
 info=HfApi().get_paths_info('yifengzhu-hf/LIBERO-datasets',[name],repo_type='dataset',revision=REV)[0]
 p=Path(hf_hub_download('yifengzhu-hf/LIBERO-datasets',name,repo_type='dataset',revision=REV,local_dir=ROOT/'native'))
 h=digest(p,'sha256').hexdigest()
 expected=info.lfs.sha256 if hasattr(info.lfs,'sha256') else info.lfs['sha256']
 assert h==expected,(task,'LFS sha256 mismatch')
 record=dict(task=task,path=str(p),revision=REV,sha256=h,bytes=p.stat().st_size)
 (ROOT/f'{task}_source.json').write_text(json.dumps(record,indent=2))
 print('NATIVE_VERIFIED',task,record,flush=True)

def official():
 prefix='checkpoints/pi0_base/'; objects=[];page=''
 while True:
  url='https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix='+prefix+('&pageToken='+page if page else '')
  with urllib.request.urlopen(url,timeout=60) as r:d=json.load(r)
  objects.extend(d.get('items',[]));page=d.get('nextPageToken')
  if not page:break
 assert objects
 (ROOT/'base_inventory.json').write_text(json.dumps(objects,indent=2))
 def one(o):
  rel=o['name'][len(prefix):]
  if not rel or rel.endswith('/'):return
  p=ROOT/'pi0_base'/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists() or p.stat().st_size!=int(o['size']):
   tmp=p.with_name(p.name+'.partial')
   urllib.request.urlretrieve('https://storage.googleapis.com/openpi-assets/'+o['name'],tmp)
   assert tmp.stat().st_size==int(o['size'])
   tmp.replace(p)
  assert base64.b64encode(digest(p,'md5').digest()).decode()==o['md5Hash']
  print('OFFICIAL_OBJECT_VERIFIED',rel,int(o['size']),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(one,objects))
 (ROOT/'official_base_verified.json').write_text(json.dumps(dict(uri='gs://openpi-assets/checkpoints/pi0_base',objects=len(objects))))
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(lambda t: official() if t=='base' else native(t),['D0','D1','base']))
