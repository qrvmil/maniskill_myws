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
    magnitudes=[]; executed_magnitudes=[]; clipped=[]; diagnostics=[]
    env_seconds = 0.
    inference_seconds = initial_inference_seconds
    while True:
        # residual returns UNIT correction; existing SAC select_delta already
        # scales, so its adapter must divide by scale exactly once.
        unit_delta = np.zeros(7, np.float32) if residual is None else residual(obs, base)
        action = contract.compose(base, unit_delta)
        magnitudes.append(np.abs(residual_scale*np.asarray(unit_delta)))
        executed_magnitudes.append(np.abs(action-base))
        detail=dict(getattr(residual,'last_diagnostics',{}))
        clipped.append(float(detail.get('otf_selected_clipped',np.any(np.abs(base+residual_scale*np.asarray(unit_delta))>1))))
        if detail:diagnostics.append(detail)
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
    row.update(residual_mean_abs=float(np.mean(magnitudes)),residual_max_abs=float(np.max(magnitudes)),
               residual_per_dim_abs=np.mean(magnitudes,axis=0).tolist(),
               executed_residual_mean_abs=float(np.mean(executed_magnitudes)),
               clipped_action_fraction=float(np.mean(clipped)))
    if diagnostics:
        row['policy_diagnostics']={k:float(np.mean([d[k] for d in diagnostics if k in d])) for k in set().union(*diagnostics)}
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


class ResidualPolicy:
    """One deployment policy shared by training and evaluation; private RNG stream."""
    def __init__(self,agent,config,mode,seed):
        import torch
        if mode not in ('deterministic_actor','otf','stochastic_actor'):
            raise ValueError(f'Unknown residual policy mode: {mode}')
        if mode=='otf' and config['otf_rollout_actions']<1:
            raise ValueError('OTF evaluation requires a positive training candidate count')
        self.agent,self.config,self.mode=agent,config,mode
        self.devices=[agent.device.index or 0] if agent.device.type=='cuda' else []
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            self.cpu_rng=torch.get_rng_state()
            self.gpu_rng=torch.cuda.get_rng_state(agent.device) if self.devices else None
        self.last_diagnostics={}

    def __call__(self,obs,base):
        import torch
        a=self.agent
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu_rng)
            if self.devices:torch.cuda.set_rng_state(self.gpu_rng,a.device)
            if self.mode=='otf':
                action=a.select_action_otf(obs['state'],base,images=obs['images'],n_actions=self.config['otf_rollout_actions'])
                delta=action-base
                diagnostics=dict(a.last_otf_metrics)
            else:
                delta=a.select_delta(obs['state'],base,images=obs['images'],deterministic=self.mode=='deterministic_actor')
                diagnostics={}
            self.cpu_rng=torch.get_rng_state()
            self.gpu_rng=torch.cuda.get_rng_state(a.device) if self.devices else None
            # Diagnostic samples do not change the rollout stream.
            with torch.no_grad():
                state=torch.as_tensor(obs['state'],dtype=torch.float32,device=a.device)[None]
                b=torch.as_tensor(base,dtype=torch.float32,device=a.device)[None]
                im=a._images_to_tensor(obs['images'][None] if obs['images'] is not None else None)
                mean,_=a.actor.sample(state,b,im,deterministic=True)
                sampled,logp=a.actor.sample(state,b,im)
                actions=torch.stack([b[0],a._clip_action(b+sampled)[0],a._clip_action(b+mean)[0]],dim=0)[None]
                q=torch.minimum(a._q_for_action_set(a.q1,state,actions,im),a._q_for_action_set(a.q2,state,actions,im))[0]
                diagnostics.update(q_base=float(q[0]),q_mean_residual=float(q[2]),alpha=float(a.alpha))
                diagnostics.setdefault('q_sampled_residual',float(q[1]))
                diagnostics.setdefault('entropy',float(-logp.mean()))
        self.last_diagnostics=diagnostics
        return np.clip(delta/a.config.action_scale,-1,1)


def residual_policy(agent,config,mode,*,seed):
    return ResidualPolicy(agent,config,mode,seed)
