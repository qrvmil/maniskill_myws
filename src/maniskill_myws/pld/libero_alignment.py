"""Seen-only alignment helpers wrapping official OpenPI/LeRobot APIs."""
import dataclasses
import json
from pathlib import Path
import numpy as np
from .libero_protocol import Protocol, file_sha256, task_key, directory_manifest, verify_directory

ALIGNMENT_DATA_VERSION='native_post_action_shift_v1'


def native_observation_action_indices(length):
    """Native create_dataset.py records obs after action: use next action label."""
    if length<2:raise ValueError('Native trajectory needs at least two frames')
    return zip(range(length-1),range(1,length),strict=True)


def audit_source_h5(path, config):
    import h5py
    source=task_key(config['source'])
    with h5py.File(path,'r') as f:
        root=f['data']
        if not str(root.attrs.get('bddl_file_name','')).endswith(source+'.bddl'):
            raise ValueError('Demonstration BDDL must match the source task')
        names=sorted(root.keys(),key=lambda s:int(s.split('_')[-1]))
        lengths=[]
        fingerprints=[]
        observation_fingerprints=[]
        for name in names:
            actions=root[name]['actions'][:]
            if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all() or np.max(np.abs(actions))>1:
                raise ValueError(f'Invalid source actions in {name}')
            lengths.append(len(actions))
            fingerprints.append(root[name]['states'][0].tolist())
            observation_fingerprints.append(root[name]['states'][1].tolist())
    return {'path':str(Path(path).resolve()),'sha256':file_sha256(path),'task':source,
            'num_demonstrations':len(names),'transitions':sum(lengths),
            'episode_names':names,'episode_lengths':lengths,'initial_sim_states':fingerprints,
            'initial_observation_sim_states':observation_fingerprints}


def convert_source_h5(path, config, *, repo_id, root):
    """Raw native HDF5 → same LeRobot feature schema as OpenPI's example.

    Both raw camera arrays are rotated exactly as the runtime adapter. No unseen
    file is opened. Raw native demos have 128px cameras; this is recorded.
    """
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    audit=audit_source_h5(path,config)
    validate_dataset_location(repo_id,root)
    with h5py.File(path,'r') as f:
        data=f['data']
        example=data[audit['episode_names'][0]]['obs']
        shape=tuple(example['agentview_rgb'].shape[1:])
        ds=LeRobotDataset.create(repo_id=repo_id,root=Path(root),robot_type='panda',fps=20,
            features={'image':{'dtype':'image','shape':shape,'names':['height','width','channel']},
                      'wrist_image':{'dtype':'image','shape':shape,'names':['height','width','channel']},
                      'state':{'dtype':'float32','shape':(8,),'names':['state']},
                      'actions':{'dtype':'float32','shape':(7,),'names':['actions']}},
            image_writer_threads=4,image_writer_processes=0)
        prompt=config['source']['name'].replace('_',' ')
        for name in audit['episode_names']:
            g=data[name]; o=g['obs']
            for i,action_index in native_observation_action_indices(len(g['actions'])):
                action=g['actions'][action_index]
                state=np.concatenate([o['ee_pos'][i],o['ee_ori'][i],o['gripper_states'][i]]).astype(np.float32)
                ds.add_frame({'image':np.ascontiguousarray(o['agentview_rgb'][i][::-1,::-1]),
                              'wrist_image':np.ascontiguousarray(o['eye_in_hand_rgb'][i][::-1,::-1]),
                              'state':state,'actions':np.asarray(action,np.float32),'task':prompt})
            ds.save_episode()
        ds.stop_image_writer()
    audit.update(repo_id=repo_id,root=str(Path(root).resolve()),camera_shape=list(shape),
                 camera_transform='rotate180 both raw native cameras; matching runtime',fps=20,
                 no_op_filter=False,split_hash=Protocol(config).split_hash,
                 alignment_data_version=ALIGNMENT_DATA_VERSION,
                 observation_action_alignment='native obs[i] is after action[i]; pair with action[i+1], drop last obs',
                 raw_transitions=audit['transitions'],transitions=audit['transitions']-audit['num_demonstrations'],
                 converted_episode_lengths=[n-1 for n in audit['episode_lengths']])
    audit['dataset_files']=directory_manifest(root,exclude=('source_audit.json',))
    return audit


def make_openpi_config(protocol_config, *, repo_id, workdir, method='full', steps=3000):
    from openpi.training import config as oc
    if method not in ('full','full_cpu','full_torch','lora','lora32'):
        raise ValueError(f'Unknown alignment method: {method}')
    base=oc.get_config('pi0_libero' if method in ('full','full_cpu','full_torch') else 'pi0_libero_low_mem_finetune')
    if method=='lora32':
        configure_lora_rank32()
    name='pi0_libero_seen_'+method
    return dataclasses.replace(base,name=name,exp_name='EXP-001',
        data=dataclasses.replace(base.data,repo_id=repo_id,extra_delta_transform=False),
        batch_size=int(protocol_config.get('sft_batch_size',1)),num_workers=0,num_train_steps=steps,ema_decay=None,seed=protocol_config['training_seed'],
        wandb_enabled=False,save_interval=int(protocol_config.get('sft_save_interval',500)),keep_period=int(protocol_config.get('sft_save_interval',500)),log_interval=10,
        assets_base_dir=str(Path(workdir)/'assets'),checkpoint_base_dir=str(Path(workdir)/'checkpoints'))


def validate_dataset_location(repo_id, root):
    parts=Path(repo_id).parts
    if len(parts)!=2 or any(p in ('.','..') for p in parts):
        raise ValueError('Dataset repo_id must be owner/name')
    root=Path(root).resolve()
    if root != root.parents[1]/repo_id:
        raise ValueError('Explicit dataset root must match the OpenPI repo_id path')


def verify_dataset_binding(audit, *, repo_id, root):
    root=Path(root).resolve()
    validate_dataset_location(repo_id,root)
    if audit.get('repo_id')!=repo_id or Path(audit.get('root','')).resolve()!=root:
        raise ValueError('Dataset repo_id/root differs from the source-only audit')
    verify_directory(root,audit.get('dataset_files',{}),exclude=('source_audit.json',))


def configure_lora_rank32():
    """PLD C.2 rank32, overriding pinned OpenPI's VLM rank16 default."""
    from openpi.models import gemma
    if getattr(gemma.get_config,'pld_rank32',False):return
    original=gemma.get_config
    def get_config(variant):
        config=original(variant)
        if variant in ('gemma_2b_lora','gemma_300m_lora'):
            config=dataclasses.replace(config,lora_configs={
                k:dataclasses.replace(v,rank=32,alpha=32.) for k,v in config.lora_configs.items()})
        return config
    get_config.pld_rank32=True
    gemma.get_config=get_config
