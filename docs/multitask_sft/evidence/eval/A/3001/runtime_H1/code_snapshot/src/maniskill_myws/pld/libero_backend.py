"""LIBERO contracts; no simulator imports until environment creation.

Actions here are robosuite normalized controller inputs, after OpenPI's
inverse normalization. OpenPI model normalization must never be applied twice.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from .state import prepare_rgb_image

CAMERA_KEYS = ('agentview_image', 'robot0_eye_in_hand_image')


def quat_to_axisangle(quat):
    q = np.array(quat, dtype=np.float64, copy=True)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Expected finite xyzw quaternion (4,)')
    w = np.clip(q[3], -1., 1.)
    den = np.sqrt(1. - w*w)
    return np.zeros(3, np.float32) if den < 1e-8 else (q[:3]*2*np.arccos(w)/den).astype(np.float32)


def convert_observation(raw, prompt, *, image_size=128):
    state = np.concatenate((raw['robot0_eef_pos'], quat_to_axisangle(raw['robot0_eef_quat']),
                            raw['robot0_gripper_qpos'])).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f'Expected finite 8D LIBERO proprioception, got {state.shape}')
    images = np.stack([prepare_rgb_image(np.asarray(raw[k])[::-1, ::-1], image_size)
                       for k in CAMERA_KEYS])
    return {'state': state, 'images': images, 'prompt': str(prompt)}


def openpi_observation(raw, prompt):
    from openpi_client import image_tools
    obs = convert_observation(raw, prompt, image_size=None)
    return {'observation/state': obs['state'], 'prompt': obs['prompt'],
            'observation/image': image_tools.resize_with_pad(obs['images'][0], 224, 224),
            'observation/wrist_image': image_tools.resize_with_pad(obs['images'][1], 224, 224)}


@dataclass(frozen=True)
class ActionContract:
    residual_scale: float = .5

    def __post_init__(self):
        if not 0 < self.residual_scale <= 1:
            raise ValueError('residual_scale must be finite in (0,1]')

    @staticmethod
    def _validate(action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (7,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
            raise ValueError(f'Expected finite bounded [-1,1] action (7,), got {a}')
        return a.copy()

    def from_env(self, action):
        """Identity in normalized OSC controller units, not model z-scores."""
        return self._validate(action)

    def to_env(self, action):
        return self._validate(action)

    def compose(self, base, unit_residual):
        a, d = self._validate(base), self._validate(unit_residual)
        return np.clip(a + self.residual_scale*d, -1, 1).astype(np.float32)


class ChunkedBasePolicy:
    """Model must implement infer(obs) and reset(seed) with isolated inference RNG.

    Cache holds post-inverse-normalization, bounded environment actions. Only
    act() consumes an action; peek() does not advance. Output clipping is explicit
    and counted, including for base-only evaluation.
    """
    def __init__(self, model, *, replan_steps=5):
        if replan_steps < 1:
            raise ValueError('replan_steps must be positive')
        self.model, self.replan_steps = model, int(replan_steps)
        self.queue = deque()
        self.clipped_components = 0

    def reset(self, seed):
        self.queue.clear()
        self.clipped_components = 0
        self.model.reset(int(seed))

    def peek(self, obs):
        if not self.queue:
            a = np.asarray(self.model.infer(obs)['actions'], dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != 7 or len(a) < self.replan_steps or not np.isfinite(a).all():
                raise ValueError(f'Expected finite chunk [H>=replan_steps,7], got {a.shape}')
            a = a[:self.replan_steps]
            self.clipped_components += int(np.count_nonzero(np.abs(a) > 1))
            self.queue.extend(np.clip(a, -1, 1))
        return self.queue[0].copy()

    def act(self, obs):
        self.peek(obs)
        return self.queue.popleft().copy()


class LiberoEnv:
    """One offscreen environment with generated-seed or explicit-state reset.

    Success and time limit are distinguished. The finite-horizon RL caller may
    combine them for its backup mask. No automatic reset hides terminal obs.
    """
    def __init__(self, task, *, render_size=256, settle_steps=10):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[task['suite']](task_order_index=0)
        matches = [(i, t) for i,t in enumerate(suite.tasks) if t.name == task['name']]
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous LIBERO task: {task}')
        self.task_id, self.task = matches[0]
        self.task_spec = task
        self.prompt = self.task.language
        self.horizon = int(task['horizon'])
        self.settle_steps = int(settle_steps)
        bddl = Path(get_libero_path('bddl_files'))/self.task.problem_folder/self.task.bddl_file
        if task.get('bddl_sha256') and hashlib.sha256(bddl.read_bytes()).hexdigest() != task['bddl_sha256']:
            raise ValueError('LIBERO BDDL differs from frozen protocol')
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=render_size,
                                     camera_widths=render_size, hard_reset=False, horizon=self.horizon+settle_steps+1)
        # EGL multisample resolves varied by 1–2 uint8 values across identical
        # physics states. Disable MSAA before any recorded/policy observation.
        sim=self.env.env.sim
        sim.model.vis.quality.offsamples=0
        context=sim._render_context_offscreen
        context.con.free()
        context._set_mujoco_context_and_buffers()
        low, high = self.env.env.action_spec
        if np.shape(low) != (7,) or not np.all(low == -1) or not np.all(high == 1):
            raise ValueError('LIBERO OSC action bounds differ from contract')
        self.contract = ActionContract()
        self.done = True

    def reset(self, *, seed, initial_state=None):
        self.env.seed(int(seed))
        obs = self.env.reset()
        if initial_state is not None:
            obs = self.env.set_init_state(initial_state)
        sampled_state = np.asarray(self.env.get_sim_state()).copy()
        for _ in range(self.settle_steps):
            obs, _, _, _ = self.env.step([0.]*6+[-1.])
        self.steps, self.done = 0, False
        state = np.asarray(self.env.get_sim_state()).copy()
        return obs, {'seed':int(seed), 'initial_state':state, 'sampled_initial_state':sampled_state,
                     'reset_hash':hashlib.sha256(state.tobytes()).hexdigest(),
                     'task_id':self.task_id, 'task_name':self.task.name}

    def step(self, action):
        if self.done:
            raise RuntimeError('Must reset before stepping a terminated environment')
        obs, _, _, info = self.env.step(self.contract.to_env(action).tolist())
        self.steps += 1
        success = bool(self.env.check_success())
        truncated = self.steps >= self.horizon and not success
        self.done = success or truncated
        return obs, float(success), success, truncated, {**info, 'success':success}

    def physics_state(self):
        return np.asarray(self.env.get_sim_state()).copy()

    def close(self):
        self.env.close()
