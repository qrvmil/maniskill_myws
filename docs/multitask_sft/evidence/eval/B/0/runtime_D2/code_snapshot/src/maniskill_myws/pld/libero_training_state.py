"""Immutable episode-boundary snapshots of replay and learner/rollout RNG."""
import json
from pathlib import Path
import numpy as np
import torch
from .libero_artifacts import write_json
from .libero_protocol import file_sha256


def save_training_state(folder, checkpoint, replay, residual, probe_rng, counters):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(checkpoint).resolve()
    replay.save(folder/'online.npz',kind='V3_resume_replay')
    payload=dict(counters=counters,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng={i:torch.cuda.get_rng_state(i) for i in residual.devices},
        residual_cpu_rng=residual.cpu_rng,residual_gpu_rng=residual.gpu_rng,
        probe_rng=probe_rng.bit_generator.state)
    torch.save(payload,folder/'state.pt')
    write_json(folder/'manifest.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        files={name:file_sha256(folder/name) for name in ('online.npz','state.pt')}))


def training_state_manifest(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if file_sha256(manifest['checkpoint'])!=manifest['checkpoint_sha256']:
        raise ValueError('Resume checkpoint changed')
    for name,sha in manifest['files'].items():
        if name not in ('online.npz','state.pt') or file_sha256(folder/name)!=sha:
            raise ValueError('Resume state changed')
    if set(manifest['files'])!={'online.npz','state.pt'}:raise ValueError('Incomplete resume snapshot')
    return manifest


def restore_training_state(folder,replay,residual,probe_rng):
    folder=Path(folder);training_state_manifest(folder)
    replay.load(folder/'online.npz')
    payload=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
    np.random.set_state(payload['numpy_rng']);torch.set_rng_state(payload['torch_rng'])
    for device,state in payload['cuda_rng'].items():torch.cuda.set_rng_state(state,device)
    residual.cpu_rng=payload['residual_cpu_rng'];residual.gpu_rng=payload['residual_gpu_rng']
    probe_rng.bit_generator.state=payload['probe_rng']
    return payload['counters']
