"""D0-only one-action intervention diagnostic with exact prefix reconstruction.

Empirical values continue with the frozen BASE after one edit. They are not
unbiased estimates of Q under the residual policy; ties are reported/excluded.
Replaying from reset avoids incomplete MuJoCo/controller/cache snapshots.
"""
import hashlib
import numpy as np
from .libero_backend import ActionContract, convert_observation


def branch_rollout(env, base, *, seed, prefix, candidate, image_size=128, gamma=.99):
    raw, info = env.reset(seed=seed)
    base.reset(seed)
    prefix_hash=hashlib.sha256(); trajectory_hash=hashlib.sha256()
    for action in prefix:
        base.act(raw)  # advance exactly once, including model RNG and chunk cache
        raw, _, terminated, truncated, _ = env.step(action)
        prefix_hash.update(env.physics_state().tobytes())
        if terminated or truncated:
            raise ValueError('Counterfactual prefix reaches terminal state')
    branch_state=env.physics_state().tolist()
    obs=convert_observation(raw,env.prompt,image_size=image_size)
    base_action=base.act(raw)
    action=ActionContract().to_env(candidate(obs,base_action.copy()))
    first_action=action.copy(); value=0.; length=0
    while True:
        raw,reward,terminated,truncated,step_info=env.step(action)
        value+=gamma**length*reward;length+=1
        trajectory_hash.update(env.physics_state().tobytes());trajectory_hash.update(action.tobytes())
        if terminated or truncated:break
        action=base.act(raw)
    return dict(reset_hash=info['reset_hash'],prefix_hash=prefix_hash.hexdigest(),branch_state=branch_state,
        trajectory_hash=trajectory_hash.hexdigest(),empirical_return=float(value),success=bool(step_info['success']),
        length=length,action=first_action.tolist(),base_action=base_action.tolist())


def ranking_summary(rows, *, tolerance=1e-6):
    correct=[];fp=[];fn=[];margins=[];q_all=[];return_all=[];empirical_ties=0
    per_state=[]
    for row in rows:
        previous=(len(correct),len(fp),len(fn))
        q=np.asarray(row['q']);r=np.asarray(row['returns'])
        if q.shape!=r.shape or q.ndim!=1 or len(q)<2 or not np.isfinite(q).all() or not np.isfinite(r).all():
            raise ValueError('Invalid counterfactual values')
        q_all.extend(q);return_all.extend(r)
        margins.extend((q[1:]-q[0]).tolist())
        for i in range(len(q)):
            for j in range(i):
                if abs(r[i]-r[j])<=tolerance:empirical_ties+=1;continue
                correct.append(float((q[i]-q[j])*(r[i]-r[j])>0))
        for i in range(1,len(q)):
            if r[i]<r[0]-tolerance:fp.append(float(q[i]>q[0]))
            if r[i]>r[0]+tolerance:fn.append(float(q[i]<=q[0]))
        per_state.append([(sum(values[start:]),len(values)-start) for values,start in zip((correct,fp,fn),previous)])
    confidence={}
    if per_state:
        counts=np.asarray(per_state,dtype=float)
        ix=np.random.default_rng(0).integers(0,len(rows),size=(2000,len(rows)))
        totals=counts[ix].sum(axis=1)
        for i,key in enumerate(('pairwise_ranking_accuracy','false_positive_rate','false_negative_rate')):
            valid=totals[:,i,1]>0
            confidence[key]=np.quantile(totals[valid,i,0]/totals[valid,i,1],[.025,.975]).tolist() if valid.any() else None
    mean=lambda x:float(np.mean(x)) if len(x) else None
    return dict(states=len(rows),state_bootstrap_95ci=confidence,non_tied_pairs=len(correct),empirical_tied_pairs=empirical_ties,
        pairwise_ranking_accuracy=mean(correct),false_positive_rate=mean(fp),empirically_worse_edits=len(fp),
        false_negative_rate=mean(fn),empirically_better_edits=len(fn),mean_q_edit_minus_base=mean(margins),
        q_margin_quantiles=np.quantile(margins,[0,.25,.5,.75,1]).tolist() if margins else [],
        mean_q=mean(q_all),mean_empirical_return=mean(return_all),
        return_mse=mean((np.asarray(q_all)-return_all)**2),
        limitation='One-action intervention then base continuation; Q learns combined-policy continuation. Ties excluded from accuracy; FPR conditional on empirically worse edits, FNR on better edits.')
