"""Download pinned official RLDS suites and extract exact training tasks only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME', '/workspace/.hf_home')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
from data import WORK, DATA_REPO, DATA_REV, TRAIN_SETS, TASKS, training_tasks, task_for_instruction, compute_stats, validate_demo_counts


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare():
    from huggingface_hub import snapshot_download
    import h5py
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    suites = ['libero_spatial_no_noops', 'libero_goal_no_noops']
    source = Path(snapshot_download(DATA_REPO, repo_type='dataset', revision=DATA_REV,
        allow_patterns=[s+'/**' for s in suites], local_dir=WORK/'rlds', max_workers=4))
    (WORK/'data').mkdir(parents=True, exist_ok=True)
    handles, counts, lengths = {}, {t:0 for t in ('D0','D1','D2')}, {t:[] for t in ('D0','D1','D2')}
    try:
        for t in counts:
            handles[t] = h5py.File(WORK/'data'/f'{t}.hdf5.tmp', 'w')
            handles[t].attrs['task'] = t
            handles[t].attrs['instruction'] = TASKS[t]['name'].replace('_', ' ')
        for suite in suites:
            builder = tfds.builder_from_directory(str(source/suite/'1.0.0'))
            for ep in tfds.as_numpy(builder.as_dataset(split='train')):
                steps = list(ep['steps'])
                if not steps:
                    raise ValueError('Empty RLDS episode')
                t = task_for_instruction(steps[0]['language_instruction'])
                if t is None:
                    continue
                if any(task_for_instruction(step['language_instruction']) != t for step in steps):
                    raise ValueError('Instruction changes inside episode')
                if TASKS[t]['suite']+'_no_noops' != suite:
                    raise ValueError('Task/suite mismatch')
                group = handles[t].create_group(f'demo_{counts[t]:03d}')
                for key, values in {
                    'images':np.stack([[s['observation']['image'],s['observation']['wrist_image']] for s in steps]),
                    'states':np.stack([s['observation']['state'] for s in steps]).astype(np.float32),
                    'actions':np.stack([s['action'] for s in steps]).astype(np.float32),
                }.items():
                    if key == 'images' and (values.shape[1:] != (2,256,256,3) or values.dtype != np.uint8):
                        raise ValueError(f'Unexpected images {values.shape} {values.dtype}')
                    if key != 'images' and (values.shape != (len(steps),8 if key=='states' else 7) or not np.isfinite(values).all()):
                        raise ValueError(f'Unexpected {key} {values.shape}')
                    if key == 'actions' and (np.max(np.abs(values)) > 1.00001 or not np.isin(values[:,6],[-1,1]).all()):
                        raise ValueError('Expected bounded actions and binary gripper')
                    group.create_dataset(key, data=values, compression='lzf', chunks=True)
                counts[t] += 1
                lengths[t].append(len(steps))
                print('EXTRACTED', t, counts[t], len(steps), flush=True)
        validate_demo_counts(counts)
    finally:
        for handle in handles.values():
            handle.close()
    hashes = {}
    for t in counts:
        path = WORK/'data'/f'{t}.hdf5'
        (WORK/'data'/f'{t}.hdf5.tmp').replace(path)
        hashes[t] = sha256(path)
    manifest = dict(source_repo=DATA_REPO, source_revision=DATA_REV, tasks=list(counts),
        demos=counts, episode_lengths=lengths, hashes=hashes,
        alignment='RLDS observation[t] -> action[t]; no additional shift',
        images='official 256px regenerated no-noop images, already rotated180; no resize')
    write_json(WORK/'data_manifest.json', manifest)
    for variant, tasks in TRAIN_SETS.items():
        training_tasks(tasks)
        states, actions = [], []
        for t in tasks:
            with h5py.File(WORK/'data'/f'{t}.hdf5', 'r') as f:
                for name in sorted(f):
                    states.append(f[name]['states'][:])
                    actions.append(f[name]['actions'][:])
        stats = compute_stats(states, actions)
        stats['training_tasks'] = tasks
        stats['source_hashes'] = {t:hashes[t] for t in tasks}
        stats['method'] = 'frame-weighted proprio mean/std and arm hard min/max; binary gripper unchanged'
        write_json(WORK/variant/'stats.json', {'train':stats})
    print('DATA_PREPARED', counts, flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
