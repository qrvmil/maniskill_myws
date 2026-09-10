#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maniskill_myws.pld.libero_selection import select_source_validation,intermediate_alignment_manifests

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['manifests','select'])
    p.add_argument('--config',default='configs/pld_libero/anchor_bowl_v2.json');p.add_argument('--output',required=True)
    p.add_argument('--alignment-manifest');p.add_argument('--evaluations',nargs='+');p.add_argument('--role',choices=['base','residual'],default='base')
    p.add_argument('--policy',choices=['otf','deterministic_actor'],default='otf');args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    if args.mode=='manifests':print(intermediate_alignment_manifests(args.alignment_manifest,args.output))
    else:
        cfg=json.loads(Path(args.config).read_text());result=select_source_validation(args.evaluations,cfg,role=args.role,policy=args.policy)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(result['selected_evaluation'])
if __name__=='__main__':main()
