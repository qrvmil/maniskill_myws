#!/usr/bin/env python3
"""Copy only inspectable lightweight evidence, never datasets/model parameter blobs."""
from pathlib import Path
import json
import shutil
import gzip
from maniskill_myws.pld.multitask_data import WORK,ROOT,DATA,repo_id
from maniskill_myws.pld.multitask_protocol import TRAIN_SETS,sha256
from maniskill_myws.pld.libero_protocol import directory_manifest
from maniskill_myws.pld.libero_artifacts import write_json
OUT=ROOT/'docs/multitask_sft/evidence'
OUT.mkdir(parents=True,exist_ok=True)

def copy(source,target):
    if not source.exists():return
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

for name in ('base_inventory.json','base_manifest.json','task_preflight.json','data_preflight.json','initial_parameter_sha256.json','gpu_pipeline_complete.json'):
    copy(WORK/name,OUT/name)
for folder in ('setup','paired_resets','parallel_validation','image_bank'):
    source=WORK/folder
    if source.exists():
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.txt','.md','.log','.npz'):
                copy(path,OUT/folder/path.relative_to(source))
for variant in TRAIN_SETS:
    for path in (WORK/variant).glob('*'):
        if path.is_file() and path.suffix in ('.json','.txt','.npz'):copy(path,OUT/'training'/variant/path.name)
    source=DATA/repo_id(variant)/'source_audit.json'
    copy(source,OUT/'training'/variant/'source_audit.json')
    binding=json.loads((WORK/variant/'binding.json').read_text())
    copy(Path(binding['normalization_path']),OUT/'training'/variant/'norm_stats.json')
for folder in ('eval','runtime'):
    source=WORK/folder
    if source.exists():
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.py','.sh','.patch','.md','.npz'):
                copy(path,OUT/folder/path.relative_to(source))
# Video publication copies retain matching sidecars. Never publish parallel-gate test clips.
video_root=ROOT/'videos/multitask_sft';video_root.mkdir(parents=True,exist_ok=True)
for source in (WORK/'eval').glob('*/3001/*/videos/*.mp4'):
    copy(source,video_root/source.name);copy(source.with_suffix('.json'),video_root/source.with_suffix('.json').name)
for name in ('multitask_setup','multitask_prepare','multitask_pipeline'):
    log=Path('/var/log')/(name+'.log')
    if log.exists():
        target=OUT/'logs'/(name+'.log.gz');target.parent.mkdir(parents=True,exist_ok=True)
        with log.open('rb') as src,gzip.open(target,'wb') as dst:shutil.copyfileobj(src,dst)
    copy(Path('/opt/supervisor-scripts')/(name+'.sh'),OUT/'supervisor'/(name+'.sh'))
    copy(Path('/etc/supervisor/conf.d')/(name+'.conf'),OUT/'supervisor'/(name+'.conf'))
write_json(OUT/'sha256_manifest.json',directory_manifest(OUT,exclude=('sha256_manifest.json',)))
print('EVIDENCE_COPIED',OUT)
