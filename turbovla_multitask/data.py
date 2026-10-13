"""Auditable own-corpus preprocessing; RLDS actions align with the same step."""
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('TURBOVLA_WORK', '/workspace/turbovla-multitask'))
CONFIG = json.loads((ROOT/'configs/pld_libero/multitask_sft.json').read_text())
TASKS = CONFIG['tasks']
TRAIN_SETS = CONFIG['train_sets']
DATA_REPO = 'openvla/modified_libero_rlds'
DATA_REV = '6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551'
HF_REPO = 'qrvmil/turbovla-multitask-generalization'


def training_tasks(tasks):
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in ('D0', 'D1', 'D2') for t in tasks):
        raise ValueError('Only unique D0/D1/D2 training tasks are allowed')
    return list(tasks)


def task_for_instruction(instruction):
    if isinstance(instruction, bytes):
        instruction = instruction.decode('utf-8')
    return next((t for t in ('D0', 'D1', 'D2') if TASKS[t]['name'].replace('_', ' ') == instruction), None)


def action_chunk(actions, index, horizon=12):
    if not 0 <= index < len(actions) or horizon < 1:
        raise ValueError('Invalid chunk index/horizon')
    valid = min(horizon, len(actions)-index)
    indices = np.minimum(np.arange(index, index+horizon), len(actions)-1)
    return actions[indices].copy(), (np.arange(horizon) < valid).astype(np.float32)


def compute_stats(states, actions):
    s, a = np.concatenate(states).astype(np.float64), np.concatenate(actions).astype(np.float64)
    if s.ndim != 2 or s.shape[1] != 8 or a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError('Expected finite 8D states and 7D actions')
    return dict(proprio=dict(mean=s.mean(0).tolist(), std=s.std(0).tolist()),
                action=dict(min=a.min(0).tolist(), max=a.max(0).tolist()), frames=len(s))


def normalize_actions(actions, stats):
    result = np.asarray(actions, dtype=np.float32).copy()
    low, high = np.asarray(stats['action']['min']), np.asarray(stats['action']['max'])
    result[..., :6] = np.clip(2*(result[..., :6]-low[:6])/(high[:6]-low[:6]+1e-6)-1, -1, 1)
    return result


def denormalize_actions(actions, stats):
    result = np.asarray(actions, dtype=np.float32).copy()
    low, high = np.asarray(stats['action']['min']), np.asarray(stats['action']['max'])
    result[..., :6] = .5*(result[..., :6]+1)*(high[:6]-low[:6])+low[:6]
    return result
