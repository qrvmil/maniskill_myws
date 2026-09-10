"""Source-replay critic diagnostics. These are not environment success metrics."""
import numpy as np
import torch


def diagnose_critic(agent,buffer,*,count=256,seed=0):
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(buffer),size=min(count,len(buffer)),replace=False)
    values={k:[] for k in ('mc_return','q_base','q_random_edit','q_mean_edit','q_sampled_edit','entropy')}
    devices=[agent.device.index or 0] if agent.device.type=='cuda' else []
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        if devices:
            torch.cuda.set_rng_state(torch.Generator(device=agent.device).manual_seed(seed).get_state(),agent.device)
        for start in range(0,len(indices),16):
            ix=indices[start:start+16]
            obs=torch.as_tensor(buffer.obs[ix],device=agent.device)
            base=torch.as_tensor(buffer.base_actions[ix],device=agent.device)
            images=agent._images_to_tensor(buffer.images[ix] if buffer.images is not None else None)
            delta,_=agent.actor.sample(obs,base,images,deterministic=True)
            sampled,logp=agent.actor.sample(obs,base,images)
            random=torch.as_tensor(rng.uniform(-agent.config.action_scale,agent.config.action_scale,size=base.shape),device=agent.device,dtype=base.dtype)
            actions=torch.stack([base,agent._clip_action(base+random),agent._clip_action(base+delta),agent._clip_action(base+sampled)],1)
            q=torch.minimum(agent._q_for_action_set(agent.q1,obs,actions,images),agent._q_for_action_set(agent.q2,obs,actions,images)).cpu().numpy()
            values['mc_return'].extend(buffer.mc_returns[ix].tolist())
            values['entropy'].extend((-logp).cpu().numpy().reshape(-1).tolist())
            for i,k in enumerate(('q_base','q_random_edit','q_mean_edit','q_sampled_edit')):values[k].extend(q[:,i].tolist())
    result={k:dict(mean=float(np.mean(v)),min=float(np.min(v)),max=float(np.max(v)),median=float(np.median(v))) for k,v in values.items()}
    result.update(states=len(indices),indices=indices.tolist(),raw=values,alpha=float(agent.alpha.detach().cpu()),
                  random_edit_preference_rate=float(np.mean(np.array(values['q_random_edit'])>values['q_base'])),
                  mean_edit_preference_rate=float(np.mean(np.array(values['q_mean_edit'])>values['q_base'])),
                  sampled_edit_preference_rate=float(np.mean(np.array(values['q_sampled_edit'])>values['q_base'])),
                  interpretation='MC is recorded base-policy return; Q bootstraps the current target policy. Their difference is not an unbiased calibration error.')
    return result
