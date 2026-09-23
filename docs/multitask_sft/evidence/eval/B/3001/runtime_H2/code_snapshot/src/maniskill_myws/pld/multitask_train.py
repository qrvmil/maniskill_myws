"""Save/sampler instrumentation around the unchanged pinned official JAX trainer."""
import hashlib
import json
import os
import numpy as np
from .multitask_protocol import BalancedSampler,TRAIN_SETS,UPDATES,sha256,require_binding
from .multitask_data import WORK,DATA,BASE,ROOT,repo_id,train_config,load_script
from .libero_artifacts import write_json
from .libero_protocol import directory_manifest,verify_directory
from .libero_sanity import completed_updates


def require_identical_initialization(actual,reference):
    if actual!=reference:raise ValueError('Independent pretrained+LoRA initial parameters differ')


def sampling_summary(indices,lengths,updates=3001,batch_size=8):
    used=np.asarray(indices[:updates*batch_size])
    if len(used)!=updates*batch_size:raise ValueError('Missing optimizer-consumed sample indices')
    task=np.searchsorted(np.cumsum(lengths),used,side='right')
    if np.any(task>=len(lengths)):raise ValueError('Sample index outside corpus')
    counts=np.bincount(task,minlength=len(lengths))
    return dict(optimizer_examples=len(used),counts=counts.tolist(),frequencies=(counts/len(used)).tolist(),
        per_batch_counts=[np.bincount(batch,minlength=len(lengths)).tolist() for batch in task.reshape(-1,batch_size)],
        fetched_examples=len(indices),lookahead_examples_excluded=len(indices)-len(used))


def train(variant):
    import jax
    import logging
    from openpi.models import gemma
    os.environ['HF_LEROBOT_HOME']=str(DATA)
    logging.basicConfig(level=logging.INFO)
    binding=json.loads((WORK/variant/'binding.json').read_text())
    require_binding(binding,variant,sha256(WORK/'base_manifest.json'))
    verify_directory(BASE,json.loads((WORK/'base_manifest.json').read_text()))
    audit_path=DATA/repo_id(variant)/'source_audit.json'
    if sha256(audit_path)!=binding['source_audit_sha256']:raise ValueError('Source audit drift')
    audit=json.loads(audit_path.read_text())
    verify_directory(audit_path.parent,audit['dataset_files'],exclude=('source_audit.json',))
    cfg=train_config(variant)
    for architecture in ('gemma_2b_lora','gemma_300m_lora'):
        for lora in gemma.get_config(architecture).lora_configs.values():
            if lora.rank!=32 or lora.alpha!=32:raise ValueError('LoRA rank/alpha drift')
    if cfg.batch_size!=8 or cfg.seed!=0 or cfg.ema_decay is not None or cfg.model.action_horizon!=50:
        raise ValueError('Scientific training recipe changed')
    trainer=load_script('train');state={};indices=[]
    dl=trainer._data_loader
    orig_torch=dl.TorchDataLoader
    lengths=binding['task_frame_counts']
    class LoggedDataset:
        def __init__(self,dataset):self.dataset=dataset
        def __len__(self):return len(self.dataset)
        def __getitem__(self,index):
            indices.append(int(index));return self.dataset[index]
    def loader(dataset,local_batch_size,**kwargs):
        if kwargs.get('num_workers',0)!=0:raise ValueError('Logging requires synchronous loader')
        if len(dataset)!=sum(lengths):raise ValueError('Task frame bounds mismatch')
        if len(lengths)>1:
            kwargs.update(sampler=BalancedSampler(lengths,(cfg.num_train_steps+1)*cfg.batch_size,seed=cfg.seed),shuffle=False)
        return orig_torch(LoggedDataset(dataset),local_batch_size,**kwargs)
    dl.TorchDataLoader=loader
    orig_dir=trainer._checkpoints.initialize_checkpoint_dir
    orig_loader=dl.create_data_loader
    orig_init=trainer.init_train_state
    orig_save=trainer._checkpoints.save_state
    def capture_dir(*args,**kwargs):
        result=orig_dir(*args,**kwargs);state['manager']=result[0];return result
    def capture_loader(*args,**kwargs):
        result=orig_loader(*args,**kwargs);state['loader']=result;return result
    def record_save(train_state,updates,loop_index):
        orig_save(state['manager'],train_state,state['loader'],updates)
        state['manager'].wait_until_finished()
        checkpoint=cfg.checkpoint_dir/str(updates)
        norm=checkpoint/'assets'/repo_id(variant)/'norm_stats.json'
        if sha256(norm)!=binding['normalization_sha256']:raise ValueError('Saved normalization drift')
        write_json(WORK/variant/f'update_{updates}.json',dict(variant=variant,optimizer_updates=updates,
            upstream_loop_index=loop_index,checkpoint=str(checkpoint),normalization_sha256=sha256(norm),
            params_files=directory_manifest(checkpoint/'params')))
        if updates:
            stats=sampling_summary(indices,lengths,updates,cfg.batch_size)
            write_json(WORK/variant/f'sampling_{updates}.json',stats)
        print('MULTITASK_SAVED',variant,updates,flush=True)
    def capture_initial(*args,**kwargs):
        result=orig_init(*args,**kwargs);initial=result[0];jax.block_until_ready(initial)
        if int(initial.step)!=0:raise ValueError('Nonzero starting optimizer counter')
        digest=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(initial.params)[0]:
            array=np.asarray(value)
            digest.update(str(path).encode());digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode());digest.update(array.tobytes())
        signature=digest.hexdigest();reference=WORK/'initial_parameter_sha256.json'
        if reference.exists():require_identical_initialization(signature,json.loads(reference.read_text())['sha256'])
        else:
            if variant!='A':raise ValueError('A must establish initialization reference first')
            write_json(reference,dict(sha256=signature,seed=0,base_manifest_sha256=binding['base_manifest_sha256']))
        write_json(WORK/variant/'initialization.json',dict(parameter_sha256=signature,independent_base_load=True,
            base_manifest_sha256=binding['base_manifest_sha256'],functional_lora_perturbation=True))
        record_save(initial,0,None);return result
    def save(manager,train_state,data_loader,loop_index):
        updates=completed_updates(loop_index,int(train_state.step))
        if updates in UPDATES:record_save(train_state,updates,loop_index)
    trainer._checkpoints.initialize_checkpoint_dir=capture_dir
    dl.create_data_loader=capture_loader
    trainer.init_train_state=capture_initial
    trainer._checkpoints.save_state=save
    trainer.main(cfg)
    stats=sampling_summary(indices,lengths)
    if np.max(np.abs(np.asarray(stats['frequencies'])-1/len(lengths)))>.02:
        raise ValueError('Realized optimizer task frequencies outside preregistered 2pp tolerance')
    np.savez_compressed(WORK/variant/'optimizer_sample_indices.npz',indices=indices[:3001*8])
    write_json(WORK/variant/'sampling_final.json',stats)
    write_json(WORK/variant/'training_complete.json',dict(variant=variant,updates=3001,checkpoints=list(UPDATES)))
