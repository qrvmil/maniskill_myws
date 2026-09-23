#!/usr/bin/env python3
"""Check real converted samples against native cameras, state, next-action labels and prompt."""
import json
import os
import h5py
import numpy as np
from maniskill_myws.pld.multitask_data import WORK,DATA,repo_id,train_config
from maniskill_myws.pld.multitask_protocol import TRAIN_SETS,prompt
from maniskill_myws.pld.libero_artifacts import write_json
os.environ['HF_LEROBOT_HOME']=str(DATA)
from openpi.training.data_loader import create_torch_dataset
checks=[]
for variant,tasks in TRAIN_SETS.items():
    cfg=train_config(variant);dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    audit=json.loads((DATA/repo_id(variant)/'source_audit.json').read_text())
    offset=0
    for task in tasks:
        source=audit['sources'][task]
        with h5py.File(source['path'],'r') as h:
            for demo,length in zip(source['demo_names'],source['aligned_lengths']):
                for frame in (0,length-1):
                    item=ds[offset+frame];native=h['data'][demo];obs=native['obs']
                    assert item['prompt']==prompt(task)
                    assert np.asarray(item['actions']).shape==(50,7)
                    np.testing.assert_array_equal(item['actions'][0],np.asarray(native['actions'][frame+1],np.float32))
                    expected=np.concatenate([obs['ee_pos'][frame],obs['ee_ori'][frame],obs['gripper_states'][frame]]).astype(np.float32)
                    np.testing.assert_array_equal(item['state'],expected)
                    for camera,native_key in [('image','agentview_rgb'),('wrist_image','eye_in_hand_rgb')]:
                        expected=obs[native_key][frame][::-1,::-1].transpose(2,0,1)/255.
                        np.testing.assert_allclose(item[camera],expected,rtol=0,atol=3e-8)
                    checks.append(dict(variant=variant,task=task,demo=demo,frame=frame,prompt=prompt(task)))
                offset+=length
    assert offset==len(ds)
    print('REAL_DATA_PREFLIGHT_PASSED',variant,len(ds),flush=True)
write_json(WORK/'data_preflight.json',checks)
