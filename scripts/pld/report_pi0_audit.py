#!/usr/bin/env python3
"""Validate complete paired audit evidence, export tables, and draw required figures."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_sanity import UPDATES,summarize_rows,paired_change
from maniskill_myws.pld.libero_protocol import file_sha256

MODELS=[str(x) for x in UPDATES]+['lora_initialization','positive_control']
COLORS={'D0':'#24669C','D1':'#CC6B2C'}
LABELS={**{str(x):f'{x:,} D0 updates' for x in UPDATES},
        '0':'Pre-SFT (official weights + D0 statistics)',
        'lora_initialization':'LoRA initialization (0 updates)',
        'positive_control':'Official pi0.5 LIBERO (positive control)'}


def validate_model_identity(work, model, binding):
    if binding['model'] != model or binding['seed_block'] != list(range(9000, 9050)):
        raise ValueError('Evaluation model label or registered seeds differ')
    horizon = 10 if model == 'positive_control' else 50
    if binding['action_horizon'] != horizon or binding['action_dim'] != 32:
        raise ValueError('Evaluation action contract differs')
    # An archived run retains its original paths. Anchor those paths to the
    # registered initial checkpoint, not the archive's current filesystem home.
    initial = json.loads((work / 'sft/update_0.json').read_text())
    initial_path = Path(initial['checkpoint_directory'])
    suffix = ('sft', 'checkpoints', 'pi0_libero_seen_lora32', 'EXP-001', '0')
    if (initial['optimizer_updates'] != 0 or initial['upstream_loop_index'] is not None
            or not initial_path.is_absolute() or initial_path.parts[-5:] != suffix):
        raise ValueError('Invalid registered initial checkpoint record')
    registered_work = initial_path.parents[4]
    if model in ('0', 'positive_control'):
        name = 'pi0_base' if model == '0' else 'pi05_libero'
        expected = registered_work.parent / 'audit-checkpoints' / name
    else:
        updates = 0 if model == 'lora_initialization' else int(model)
        if updates not in UPDATES:
            raise ValueError('Unregistered optimizer milestone')
        expected = registered_work / 'sft/checkpoints/pi0_libero_seen_lora32/EXP-001' / str(updates)
        record = json.loads((work / f'sft/update_{updates}.json').read_text())
        if (record['optimizer_updates'] != updates
                or record['upstream_loop_index'] != (updates - 1 if updates else None)
                or Path(record['checkpoint_directory']).resolve() != expected.resolve()
                or record['normalization_sha256'] != binding['normalization_sha256']):
            raise ValueError('Checkpoint label disagrees with saved optimizer-update record')
    if Path(binding['checkpoint']).resolve() != expected.resolve():
        raise ValueError('Checkpoint path disagrees with registered model')


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect(work,out):
    rows={};summaries={};csv_rows=[];episode_csv=[];paired={}
    for model in MODELS:
        folder=work/'eval'/model
        complete=json.loads((folder/'complete.json').read_text())
        prompts=json.loads((folder/'prompt_evidence.json').read_text())
        binding=json.loads((folder/'binding.json').read_text())
        if complete['model']!=model:raise ValueError('Model identity mismatch')
        validate_model_identity(work, model, binding)
        if model not in ('0','positive_control'):
            if file_sha256(folder/'checkpoint_params_sha256.json')!=binding['checkpoint_params_manifest_sha256']:
                raise ValueError('Learned checkpoint content manifest changed')
        if sum(x['calls'] for x in prompts.values())!=complete['inference_calls']:
            raise ValueError('Missing real inference prompt evidence')
        rows[model]={};summaries[model]={};paired[model]={}
        for task in ('D0','D1'):
            data=json.loads((folder/(task+'_episodes.json')).read_text())
            rows[model][task]=data
            summary=summarize_rows(data)
            saved=json.loads((folder/(task+'_summary.json')).read_text())
            for key,value in summary.items():
                if not np.allclose(value,saved[key]):raise ValueError('Summary does not reproduce episodes')
            expected=next(t['name'].replace('_',' ') for t in json.loads((ROOT/'configs/pld_libero/pi0_sanity_audit.json').read_text())['tasks'] if t['distance']==task)
            if any(r['prompt']!=expected for r in data) or expected not in prompts:
                raise ValueError('Task prompt evidence missing or wrong')
            stages=dict(reach=sum(r['ever_reached_10cm'] for r in data),
                        grasp=sum(r['task_progress']['ever_grasped'] for r in data),
                        lift=sum(r['task_progress']['ever_lifted_3cm'] for r in data),
                        success=summary['successes'])
            summary['stages']=stages;summaries[model][task]=summary
            csv_rows.append(dict(model=model,label=LABELS[model],task=task,n=summary['n'],
                successes=summary['successes'],success_rate_percent=100*summary['success_rate'],
                wilson95_low_percent=100*summary['wilson_95ci'][0],wilson95_high_percent=100*summary['wilson_95ci'][1],
                mean_episode_length=summary['mean_episode_length'],**{k+'_count':v for k,v in stages.items()}))
            for r in data:
                episode_csv.append(dict(model=model,task=task,seed=r['seed'],success=int(r['success']),
                    length=r['length'],reset_hash=r['reset_hash'],prompt=r['prompt'],
                    reach=int(r['ever_reached_10cm']),grasp=int(r['task_progress']['ever_grasped']),
                    lift=int(r['task_progress']['ever_lifted_3cm']),
                    min_reach_m=r['task_progress']['minimum_reach_distance'],
                    max_lift_m=r['task_progress']['maximum_bowl_lift']))
        if model!='positive_control':
            d0=json.loads((work/'d0_binding.json').read_text())
            if binding['normalization_sha256']!=d0['normalization_sha256']:
                raise ValueError('D0 model statistics differ')
    for model in MODELS:
        for task in ('D0','D1'):
            paired[model][task]=paired_change(rows['0'][task],rows[model][task])
    write_csv(out/'success_rates.csv',csv_rows);write_csv(out/'episodes.csv',episode_csv)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'paired_changes.json').write_text(json.dumps(paired,indent=2)+'\n')
    return summaries,paired,rows


def figures(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,
        'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240})
    def style(ax):
        ax.set_ylim(-3,105);ax.set_yticks(range(0,101,20));ax.set_ylabel('Success rate (%)')
        ax.grid(axis='y',color='#DDE1E5',linewidth=.7);ax.set_axisbelow(True)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,5.2));style(ax)
    for task,marker in [('D0','o'),('D1','s')]:
        y=np.array([summary[str(x)][task]['success_rate'] for x in UPDATES])*100
        ci=np.array([summary[str(x)][task]['wilson_95ci'] for x in UPDATES])*100
        ax.errorbar(UPDATES,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),
            color=COLORS[task],marker=marker,markersize=6,capsize=4,linewidth=2,
            linestyle='-' if task=='D0' else '--',label=task)
    ax.set_xticks(UPDATES,['0','500','1,000','2,000','3,001'])
    ax.set_xlabel('D0 SFT optimizer updates');ax.set_title('D0 learning and D1 generalization',pad=18)
    ax.legend(loc='best',frameon=False)
    fig.text(.12,-.015,'50 paired generated-reset episodes per task; pointwise Wilson 95% intervals.\nStep 0: official pi0 weights + D0 statistics. LoRA initialization is reported separately.',fontsize=9,color='#444444')
    fig.tight_layout();save(fig,'sft_trajectory')
    fig,ax=plt.subplots(figsize=(8.6,5.2));style(ax)
    order=['0','3001','positive_control'];x=np.arange(3);width=.32
    for i,task in enumerate(('D0','D1')):
        y=np.array([summary[m][task]['success_rate'] for m in order])*100
        ci=np.array([summary[m][task]['wilson_95ci'] for m in order])*100
        pos=x+(i-.5)*width
        ax.bar(pos,y,width,color=COLORS[task],label=task,alpha=.94,
            yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),capsize=4,
            error_kw={'linewidth':1,'ecolor':'#333333'})
        for j,(px,value,m) in enumerate(zip(pos,y,order)):
            # Counts near lower bar area leave upper confidence intervals unobstructed.
            ax.text(px,ci[j,1]+3 if value<=20 else value/2,f'{summary[m][task]["successes"]}/50',ha='center',va='center',
                color='white' if value>20 else '#222222',fontsize=10)
    ax.set_xticks(x,['Pre-SFT\nD0 setup','Final D0 SFT\n3,001 updates','Official pi0.5\nLIBERO control'])
    ax.set_title('Base-model comparison on D0 and D1',pad=18);ax.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.02))
    fig.text(.12,-.015,'N = 50 per model and task; error bars show pointwise Wilson 95% intervals.\nThe official LIBERO model is a positive control and uses its own training statistics.',fontsize=9,color='#444444')
    fig.tight_layout();save(fig,'positive_control_comparison')
    fig,ax=plt.subplots(figsize=(8.6,5.2));style(ax);ax.set_ylabel('Episodes reaching stage (%)')
    for key,label,color,marker in [('reach','Reach < 10 cm','#24669C','o'),('grasp','Grasp','#8A779D','s'),('lift','Lift > 3 cm','#B89934','^'),('success','Task success','#CC6B2C','D')]:
        y=[summary[str(x)]['D1']['stages'][key]*2 for x in UPDATES]
        ax.plot(UPDATES,y,label=label,color=color,marker=marker,linewidth=1.8)
    ax.set_xticks(UPDATES,['0','500','1,000','2,000','3,001']);ax.set_xlabel('D0 SFT optimizer updates')
    ax.set_title('D1 behavior stages during D0 training',pad=18);ax.legend(frameon=False)
    fig.text(.12,-.015,'N = 50 paired episodes per point. Descriptive stages are not necessarily nested.\nReach uses gripper-to-target body origin distance; grasp uses simulator contact checks.',fontsize=9,color='#444444')
    fig.tight_layout();save(fig,'d1_behavior_stages')


def action_differences(work,out):
    ref=np.load(work/'eval/0/d1_first_actions.npz')
    results=[];raw=[]
    for model in [str(x) for x in UPDATES]+['lora_initialization']:
        d=np.load(work/'eval'/model/'d1_first_actions.npz')
        for key in ('seeds','reset_hashes'):
            if not np.array_equal(d[key],ref[key]):raise ValueError('Action diagnostic states are not paired')
        for kind in ('chunks','executed'):
            delta=d[kind]-ref[kind]
            results.append(dict(model=model,action_kind=kind,n_states=10,actions_per_state=d[kind].shape[1],
                mean_l1=float(np.abs(delta).sum(-1).mean()),mean_l2=float(np.linalg.norm(delta,axis=-1).mean()),
                **{f'dim_{i}_mae':float(np.abs(delta[...,i]).mean()) for i in range(7)}))
            for j,seed in enumerate(d['seeds']):
                for t,a in enumerate(d[kind][j]):
                    raw.append(dict(model=model,kind=kind,seed=int(seed),action_index=t,**{f'action_{i}':float(v) for i,v in enumerate(a)}))
    write_csv(out/'action_differences.csv',results);write_csv(out/'first_actions.csv',raw)
    return results


def generate(work, output):
    output.mkdir(parents=True,exist_ok=True)
    summary,paired,rows=collect(work,output)
    figures(summary,output);action_differences(work,output)
    inputs=[*work.glob('eval/*/*.json'),*work.glob('eval/*/*.npz'),
            *work.glob('sft/update_*.json'),work/'training_complete.json',
            work/'normalization_audit.json',work/'d0_binding.json']
    sources={str(p.relative_to(work)):file_sha256(p) for p in inputs}
    artifacts=[*output.glob('*.png'),*output.glob('*.pdf'),*output.glob('*.csv')]
    (output/'figure_provenance.json').write_text(json.dumps(dict(sources=sources,n_per_task=50,
        seeds=list(range(9000,9050)),optimizer_updates=list(UPDATES),confidence_intervals='Wilson pointwise95%',
        outputs={p.name:file_sha256(p) for p in artifacts},
        script_sha256=file_sha256(__file__)),indent=2)+'\n')
    return summary,paired,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('/workspace/audit-run'))
    p.add_argument('--output',type=Path,default=ROOT/'docs/pi0_audit');args=p.parse_args()
    summary,_,_=generate(args.work,args.output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
