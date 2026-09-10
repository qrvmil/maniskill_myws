#!/usr/bin/env python3
"""Fetch and convert the checksum-pinned official SERL ImageNet ResNet10 trunk."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maniskill_myws.pld.libero_protocol import file_sha256
from maniskill_myws.pld.serl_encoder import SERLTrunk,convert_serl_params
import torch

SOURCE='https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl'
SHA256='175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source-pickle')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    src=Path(args.source_pickle) if args.source_pickle else out.with_suffix('.pkl')
    if not src.exists():urllib.request.urlretrieve(SOURCE,src)
    if file_sha256(src)!=SHA256:raise ValueError('Official SERL source checksum mismatch; refuse unpickling')
    with src.open('rb') as stream:params=pickle.load(stream)
    state=convert_serl_params(params)
    SERLTrunk().load_state_dict(state,strict=True)
    torch.save({'trunk':state,'source_sha256':SHA256},out)
    out.with_suffix('.json').write_text(json.dumps(dict(source=SOURCE,source_sha256=SHA256,
        converted_sha256=file_sha256(out),tensors=len(state),pretraining='ImageNet-1K, official SERL release'),indent=2)+'\n')
if __name__=='__main__':main()
