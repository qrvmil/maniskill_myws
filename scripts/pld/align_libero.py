#!/usr/bin/env python3
"""Run in pinned OpenPI venv. Only source HDF5 may enter alignment."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_alignment import convert_source_h5,make_openpi_config,verify_dataset_binding
from maniskill_myws.pld.libero_protocol import Protocol,file_sha256,directory_manifest
from maniskill_myws.pld.libero_artifacts import RunArtifacts,write_json


def load_script(name):
    path=ROOT/'third_party/openpi/scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(f'openpi_script_{name}',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','norm','train'])
    p.add_argument('--config',default='configs/pld_libero/anchor_bowl.json')
    p.add_argument('--source-h5',required=True)
    p.add_argument('--repo-id',default='local/pld_libero_bowl')
    p.add_argument('--dataset-root',default='/workspace/datasets/lerobot/local/pld_libero_bowl')
    p.add_argument('--workdir',default='outputs/pld_libero/EXP-001/alignment')
    p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['full','lora','full_cpu'],default='full')
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--pytorch-base-checkpoint')
    p.add_argument('--resume-checkpoint')
    p.add_argument('--cpu-threads',type=int,default=16)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());protocol=Protocol(cfg)
    os.environ['HF_LEROBOT_HOME']=str(Path(args.dataset_root).resolve().parents[1])
    with RunArtifacts(args.output,vars(args)) as run:
        if args.mode=='prepare':
            audit=convert_source_h5(args.source_h5,cfg,repo_id=args.repo_id,root=args.dataset_root)
            write_json(Path(args.dataset_root)/'source_audit.json',audit)
            write_json(run.path/'source_audit.json',audit)
            run.meta.update(demonstrations=audit['num_demonstrations'],transitions=audit['transitions'])
            return
        audit=json.loads((Path(args.dataset_root)/'source_audit.json').read_text())
        if audit['split_hash']!=protocol.split_hash or file_sha256(args.source_h5)!=audit['sha256']:
            raise ValueError('Dataset source/split checksum mismatch')
        verify_dataset_binding(audit,repo_id=args.repo_id,root=args.dataset_root)
        train_cfg=make_openpi_config(cfg,repo_id=args.repo_id,workdir=args.workdir,method=args.method,steps=args.steps)
        (run.path/'openpi_config.txt').write_text(repr(train_cfg))
        from openpi.training import config as oc
        oc._CONFIGS_DICT[train_cfg.name]=train_cfg
        norm_path=train_cfg.assets_dirs/args.repo_id/'norm_stats.json'
        norm_binding={'source_audit_sha256':file_sha256(Path(args.dataset_root)/'source_audit.json'),
                      'data_config':repr(train_cfg.data),'model_config':repr(train_cfg.model)}
        if args.mode=='norm':
            load_script('compute_norm_stats').main(train_cfg.name)
            write_json(norm_path.with_name('normalization_provenance.json'),
                       dict(**norm_binding,statistics_sha256=file_sha256(norm_path)))
        else:
            norm_manifest=json.loads(norm_path.with_name('normalization_provenance.json').read_text())
            if norm_manifest!=dict(**norm_binding,statistics_sha256=file_sha256(norm_path)):
                raise ValueError('Normalization producer/data/statistics mismatch')
            # Official JAX trainer, including the official freeze filter for LoRA.
            if args.method=='full_cpu':
                from maniskill_myws.pld.libero_cpu_sft import train_cpu_offload
                checkpoint=train_cpu_offload(train_cfg,args,run)
            else:
                load_script('train').main(train_cfg)
                checkpoint=train_cfg.checkpoint_dir/str(args.steps-1)
            normalizer=checkpoint/'assets'/args.repo_id/'norm_stats.json'
            if not checkpoint.exists() or not normalizer.exists():
                raise RuntimeError('Official trainer did not produce expected checkpoint and statistics')
            manifest={'training_tasks':[protocol.source],'split_hash':protocol.split_hash,
                      'pretrained_checkpoint':'gs://openpi-assets/checkpoints/pi0_base',
                      'alignment_steps':args.steps,'method':args.method,
                      'aligned_checkpoint':str(checkpoint.resolve()),
                      'demonstrations':[{'path':audit['path'],'sha256':audit['sha256']}],
                      'normalization':{'path':str(normalizer.resolve()),'sha256':file_sha256(normalizer)},
                      'repo_id':args.repo_id,'workdir':str(Path(args.workdir).resolve()),
                      'source_audit':str((Path(args.dataset_root)/'source_audit.json').resolve()),
                      'source_audit_sha256':file_sha256(Path(args.dataset_root)/'source_audit.json'),
                      'checkpoint_files':directory_manifest(checkpoint)}
            write_json(run.path/'alignment_manifest.json',manifest)
            run.meta['checkpoint']=str(checkpoint)

if __name__=='__main__':
    main()
