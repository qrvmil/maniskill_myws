"""Shared LIBERO rollout path for collection, paired evaluation and smoke tests."""
import hashlib
import time
import numpy as np
from .libero_backend import ActionContract, convert_observation


def run_episode(env, base_policy, *, seed, image_size=128, residual=None,
                residual_scale=.5, on_transition=None, demonstration_states=None):
    start = time.perf_counter()
    raw, reset_info = env.reset(seed=int(seed))
    if demonstration_states is not None:
        from .libero_protocol import assert_held_out
        assert_held_out(reset_info['sampled_initial_state'],demonstration_states)
        assert_held_out(reset_info['initial_state'],demonstration_states)
    base_policy.reset(int(seed))
    obs = convert_observation(raw, env.prompt, image_size=image_size)
    inference_start=time.perf_counter()
    base = base_policy.act(raw)
    initial_inference_seconds=time.perf_counter()-inference_start
    contract = ActionContract(residual_scale)
    transitions = []
    full_physics = []
    digest = hashlib.sha256()
    image_digest = hashlib.sha256()
    env_seconds = 0.
    inference_seconds = initial_inference_seconds
    while True:
        # residual returns UNIT correction; existing SAC select_delta already
        # scales, so its adapter must divide by scale exactly once.
        unit_delta = np.zeros(7, np.float32) if residual is None else residual(obs, base)
        action = contract.compose(base, unit_delta)
        t = time.perf_counter()
        next_raw, reward, terminated, truncated, info = env.step(action)
        env_seconds += time.perf_counter()-t
        done = bool(terminated or truncated)
        next_obs = convert_observation(next_raw, env.prompt, image_size=image_size)
        t = time.perf_counter()
        next_base = np.zeros(7, np.float32) if done else base_policy.act(next_raw)
        inference_seconds += time.perf_counter()-t
        tr = dict(obs=obs['state'], action=action, base_action=base, reward=float(reward),
                  next_obs=next_obs['state'], next_base_action=next_base, done=done,
                  images=obs['images'], next_images=next_obs['images'])
        physics = env.physics_state()
        full_physics.append(physics)
        digest.update(physics.tobytes())
        digest.update(bytes([bool(terminated),bool(truncated)]))
        image_digest.update(tr['images'].tobytes())
        for value in (tr['obs'], action, tr['next_obs']):
            digest.update(value.tobytes())
        transitions.append(tr)
        if on_transition is not None:
            on_transition(tr)
        obs, base = next_obs, next_base
        if done:
            break
    row = dict(seed=int(seed), reset_hash=reset_info['reset_hash'],
               initial_state=np.asarray(reset_info['initial_state']).tolist(),
               physics_states=[x.tolist() for x in full_physics],
               trajectory_hash=digest.hexdigest(), image_hash=image_digest.hexdigest(), length=len(transitions),
               success=bool(info['success']), episode_return=sum(t['reward'] for t in transitions),
               failure_reason=None if info['success'] else 'time_limit',
               env_step_seconds=env_seconds, inference_seconds=inference_seconds,
               wall_seconds=time.perf_counter()-start,
               base_clipped_components=base_policy.clipped_components)
    return row, transitions


def insert_trajectory(buffer, transitions, *, gamma=.99):
    running = 0.
    returns = []
    for tr in reversed(transitions):
        running = tr['reward'] + gamma*(1-float(tr['done']))*running
        returns.append(running)
    for tr, mc in zip(transitions, reversed(returns), strict=True):
        buffer.add(**tr, mc_return=mc)


def actor_residual(agent):
    def select(obs, base):
        return agent.select_delta(obs['state'], base, images=obs['images'],
                                  deterministic=True)/agent.config.action_scale
    return select
