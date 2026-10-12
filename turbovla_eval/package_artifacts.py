"""Package notebooks, reproducibility files, results and separately organized videos."""
from pathlib import Path
import zipfile
import json
from eval_utils import ROOT,file_hash,write_json

if __name__=='__main__':
    archive=ROOT.parent/'turbovla_libero_evaluation.zip'
    files=[]
    for pattern in ['*.ipynb','*.py','*.sh','requirements.txt','*.log']:
        files.extend(ROOT.glob(pattern))
    for folder in ['results','provenance','docs','preview','figures']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # Precomputed observation is small; large model weights/environments remain downloadable via setup.
    files.extend(p for p in (ROOT/'assets').glob('*') if p.is_file() and p.suffix in ['.json','.npz','.md'])
    files=sorted(set(files)-{ROOT/'results/archive.json'})
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:z.write(path,str(Path('turbovla_eval')/path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        print('Archive files:',len(z.namelist()),'size MiB:',round(archive.stat().st_size/1024**2,2))
    write_json(ROOT/'results/archive.json',dict(path=str(archive),sha256=file_hash(archive),files=len(files),size_bytes=archive.stat().st_size))
