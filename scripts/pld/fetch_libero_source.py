#!/usr/bin/env python3
"""Fetch only the configured source-task demonstrations, never an entire suite."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maniskill_myws.pld.libero_alignment import audit_source_h5
from maniskill_myws.pld.libero_protocol import Protocol

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pld_libero/anchor_bowl.json')
    p.add_argument('--output-dir',default='/workspace/datasets/libero_seen')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    Protocol(cfg).require_training_task(cfg['source'])
    source=cfg['source'];root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    name=source['name']+'_demo.hdf5';path=root/name
    url=f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/{source['suite']}/{name}"
    if not path.exists():
        temp=path.with_suffix('.download')
        urllib.request.urlretrieve(url,temp)
        audit_source_h5(temp,cfg)
        temp.replace(path)
    audit=audit_source_h5(path,cfg)
    audit['download_url']=url
    path.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['path','sha256','task','num_demonstrations','transitions']},indent=2))

if __name__=='__main__':main()
