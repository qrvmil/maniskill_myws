#!/usr/bin/env python3
"""Read-only D1 versus frozen D0 normalization diagnostic; never writes training data."""
import json
from pathlib import Path
import sys
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_protocol import file_sha256
from maniskill_myws.pld.libero_sanity import require_d0_binding


def read_demo(path, task):
    actions=[];states=[]
    with h5py.File(path,'r') as f:
        data=f['data']
        if not str(data.attrs['bddl_file_name']).endswith(task+'.bddl'):
            raise ValueError('Diagnostic HDF5 task differs')
        for name in sorted(data,key=lambda x:int(x.split('_')[-1])):
            g=data[name];obs=g['obs']
            actions.append(g['actions'][1:])
            states.append(np.concatenate([obs['ee_pos'][:-1],obs['ee_ori'][:-1],obs['gripper_states'][:-1]],axis=1))
    return dict(actions=np.concatenate(actions),state=np.concatenate(states),episodes=len(actions))


def distributions(d0,d1,norm):
    result={}
    for key in ('actions','state'):
        a=d0[key];b=d1[key]
        lo,hi=a.min(0),a.max(0)
        qlo,qhi=np.quantile(a,[.01,.99],axis=0)
        stats=norm[key]
        mean=np.array(stats['mean'])[:b.shape[1]];std=np.array(stats['std'])[:b.shape[1]]
        z=(b-mean)/(std+1e-6)
        result[key]=dict(d0_n=len(a),d1_n=len(b),d0_min=lo.tolist(),d0_max=hi.tolist(),
            d0_q01=qlo.tolist(),d0_q99=qhi.tolist(),d1_min=b.min(0).tolist(),d1_max=b.max(0).tolist(),
            d1_q01=np.quantile(b,.01,axis=0).tolist(),d1_q99=np.quantile(b,.99,axis=0).tolist(),
            d0_mean=a.mean(0).tolist(),d1_mean=b.mean(0).tolist(),normalizer_mean=mean.tolist(),normalizer_std=std.tolist(),
            d1_outside_d0_minmax_fraction=((b<lo)|(b>hi)).mean(0).tolist(),
            d1_outside_d0_q01q99_fraction=((b<qlo)|(b>qhi)).mean(0).tolist(),
            d1_abs_z_above3_fraction=(abs(z)>3).mean(0).tolist(),
            d1_abs_z_q99=np.quantile(abs(z),.99,axis=0).tolist())
    return result


def main():
    work=Path('/workspace/audit-run')
    config=json.loads((ROOT/'configs/pld_libero/pi0_sanity_audit.json').read_text())
    binding=json.loads((work/'d0_binding.json').read_text());require_d0_binding(binding,config)
    p0=Path(binding['source_h5'])
    p1=Path('/workspace/audit-data/diagnostic_d1')/(config['tasks'][1]['name']+'_demo.hdf5')
    norm=json.loads(Path(binding['normalization_path']).read_text())['norm_stats']
    d0=read_demo(p0,config['tasks'][0]['name']);d1=read_demo(p1,config['tasks'][1]['name'])
    result=dict(normalization_sha256=binding['normalization_sha256'],d0_sha256=file_sha256(p0),d1_sha256=file_sha256(p1),
        d0_episodes=d0['episodes'],d1_episodes=d1['episodes'],**distributions(d0,d1,norm),
        interpretation='Unique native obs[i]/action[i+1] pairs. D0 normalization was computed by official chunked loader; its overlap/padding weighting differs from this unique-frame diagnostic. Z scores are model normalization units; raw actions are bounded OSC inputs, not meters. Native D1 demos are diagnostic references, not verified simulator replay.')
    (work/'normalization_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    require_d0_binding(binding,config)
    print(json.dumps({k:result[k]['d1_outside_d0_minmax_fraction'] for k in ('actions','state')},indent=2))

if __name__=='__main__':main()
