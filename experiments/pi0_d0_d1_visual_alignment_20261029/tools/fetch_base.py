"""Read the alignment source's immutable GCS generations, validating original MD5."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64,hashlib,json,requests,shutil
BASE=Path('/workspace/visual-alignment-repro-20261029')
source=BASE/'pi0-d0d1-visual-alignment/configs/base_inventory.json'
objects=json.loads(source.read_text());prefix='checkpoints/pi0_base/'
shutil.copy2(source,BASE/'evidence/official-base-inventory.json')

def fetch(obj):
 relative=obj['name'][len(prefix):]
 if not relative or obj['name'].endswith('/'):return None
 path=BASE/'private/official_base'/relative;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  tmp=path.with_name(path.name+'.partial')
  with requests.get('https://storage.googleapis.com/openpi-assets/'+obj['name'],params={'generation':obj['generation']},stream=True,timeout=120) as response:
   response.raise_for_status()
   with tmp.open('wb') as f:
    for block in response.iter_content(8<<20):f.write(block)
  assert tmp.stat().st_size==int(obj['size']);tmp.replace(path)
 md5=hashlib.md5();sha=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):md5.update(block);sha.update(block)
 assert base64.b64encode(md5.digest()).decode()==obj['md5Hash'],relative
 record=dict(path=relative,generation=obj['generation'],bytes=path.stat().st_size,md5=obj['md5Hash'],sha256=sha.hexdigest())
 print('OFFICIAL_BASE_VERIFIED',relative,obj['generation'],flush=True);return record
with ThreadPoolExecutor(max_workers=6) as pool:receipts=[x for x in pool.map(fetch,objects) if x]
(BASE/'evidence/official-base-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print('OFFICIAL_BASE_READY',len(receipts),sum(x['bytes'] for x in receipts),flush=True)
