"""Frozen-base audit contracts, deliberately separate from V2/V3/V4 gates."""
from pathlib import Path
import hashlib
import numpy as np
from .libero_protocol import file_sha256, task_key

UPDATES = (0, 500, 1000, 2000, 3001)


def evaluation_seeds(config, model):
    if model not in (*UPDATES, 'positive_control', 'lora_initialization'):
        raise ValueError('Unregistered model/update')
    seeds = tuple(config['seeds'])
    if seeds != tuple(range(9000, 9050)) or tuple(config['optimizer_updates']) != UPDATES:
        raise ValueError('Audit seeds or optimizer milestones changed')
    return seeds


def completed_updates(loop_index, state_step):
    """Check the actual optimizer counter, never infer updates from a filename."""
    if int(state_step) != loop_index + 1:
        raise ValueError('Optimizer update counter disagrees with loop index')
    return int(state_step)


def require_d0_binding(binding, config):
    if (binding['training_tasks'] != [task_key(config['tasks'][0])]
            or binding['num_demonstrations'] != 50 or binding['transitions'] != 5832
            or binding['pretrained_checkpoint'] != 'gs://openpi-assets/checkpoints/pi0_base'):
        raise ValueError('D0-only training provenance mismatch')
    if file_sha256(binding['normalization_path']) != binding['normalization_sha256']:
        raise ValueError('D0 normalization changed')


def model_contract(config, model, binding):
    evaluation_seeds(config, model)
    if model == 'positive_control':
        return {**config['positive_control'], 'normalization_path': None}
    return dict(config='pi0_libero_seen_lora32', optimizer_updates=model,
                normalization_path=binding['normalization_path'], purpose='d0_trajectory',
                weight_source='official_pretrained' if model==0 else 'saved_training_state',
                use_lora=model!=0, checkpoint_updates=0 if model=='lora_initialization' else model)


class PromptCheckedPolicy:
    """Assert/log the exact prompt immediately before official tokenization.

    Wrap the actual Policy input transform, preserving its transform order,
    numerical values, sampling noise and output transforms.
    """
    def __init__(self, policy, expected_prompt):
        from openpi import transforms
        self.policy = policy
        self.expected_prompt = expected_prompt
        self.calls = 0
        self.evidence = {}
        original = policy._input_transform
        chain = list(original.transforms)
        if sum(isinstance(t, transforms.TokenizePrompt) for t in chain) != 1:
            raise ValueError('Expected exactly one official prompt tokenizer')
        def checked_input(data):
            for transform in chain:
                if isinstance(transform, transforms.TokenizePrompt):
                    prompt = data.get('prompt')
                    if prompt != self.expected_prompt():
                        raise ValueError(f'Inference prompt mismatch: {prompt!r}')
                    data = transform(data)
                    digest = hashlib.sha256(np.asarray(data['tokenized_prompt']).tobytes()).hexdigest()
                    old = self.evidence.setdefault(prompt, dict(calls=0, token_sha256=digest))
                    if old['token_sha256'] != digest:
                        raise ValueError('Tokenized prompt changed for fixed instruction')
                    old['calls'] += 1
                    self.calls += 1
                else:
                    data = transform(data)
            return data
        policy._input_transform = checked_input

    def infer(self, obs, *, noise):
        before = self.calls
        result = self.policy.infer(obs, noise=noise)
        if self.calls != before + 1:
            raise ValueError('Real inference did not traverse checked prompt path')
        return result


def wilson(successes, n):
    if not 0 <= successes <= n or n < 1:
        raise ValueError('Invalid binomial count')
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [max(0., float(center-half)), min(1., float(center+half))]


def save_selected_update(save_fn, manager, state, loader, loop_index):
    updates=completed_updates(loop_index,int(state.step))
    if updates not in UPDATES:
        return None
    save_fn(manager,state,loader,updates)
    return updates


def summarize_rows(rows):
    if [r['seed'] for r in rows]!=list(range(9000,9050)):
        raise ValueError('Need exactly the 50 ordered audit seeds')
    n=len(rows);successes=sum(bool(r['success']) for r in rows)
    return dict(n=n,successes=successes,success_rate=successes/n,
                wilson_95ci=wilson(successes,n),
                mean_episode_length=float(np.mean([r['length'] for r in rows])))


def paired_change(before,after):
    summarize_rows(before);summarize_rows(after)
    if any((a['seed'],a['reset_hash'])!=(b['seed'],b['reset_hash']) for a,b in zip(before,after,strict=True)):
        raise ValueError('Paired initial scenes differ')
    delta=np.array([int(b['success'])-int(a['success']) for a,b in zip(before,after,strict=True)])
    if before is after:
        interval=[0.,0.]
        method='identity comparison; no estimation uncertainty'
    elif np.all(delta==1) or np.all(delta==-1):
        # With all deltas at one boundary, bound its population mass via the
        # exact one-sided binomial lower bound. Remaining mass may be at the
        # opposite boundary, giving a conservative paired-mean interval.
        lower=100*(2*.05**(1/len(delta))-1)
        interval=[lower,100.] if delta[0]==1 else [-100.,-lower]
        method='exact constant-discordance 95% bound with worst-case opposite mass'
    elif np.any(delta):
        boot=np.random.default_rng(0).choice(delta,size=(10000,len(delta)),replace=True).mean(1)
        interval=(100*np.quantile(boot,[.025,.975])).tolist()
        method='paired percentile bootstrap, 10000 draws, seed0'
    else:
        bound=100*(1-.05**(1/len(delta)))
        interval=[-bound,bound]
        method='exact zero-discordance 95% bound, not a degenerate bootstrap'
    return dict(n=len(delta),gained=int(sum(delta>0)),lost=int(sum(delta<0)),
                delta_pp=float(delta.mean()*100),interval_95_pp=interval,method=method)
