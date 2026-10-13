"""Build the complete3×5 matrix only from verified15-episode rollout records."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from data import ROOT,WORK,TASKS,TRAIN_SETS
from prepare import write_json,sha256
from artifacts import verified_final
sys.path.insert(0,str(ROOT/'turbovla_eval'))
from eval_utils import summarize


def summarize_cell(rows,variant,task):
    if len(rows)!=15: raise ValueError('Every matrix cell requires exactly15 episodes')
    for i,row in enumerate(rows):
        if (row.get('variant')!=variant or row.get('task')!=task or row.get('seed')!=10000+i
            or row.get('episode')!=i or type(row.get('success')) is not bool
            or row.get('seen_during_training') is not (task in TRAIN_SETS[variant])):
            raise ValueError('Episode identity, seed, success type or seen/unseen label differs')
        if row.get('horizon')!=TASKS[task]['horizon'] or not 1<=row.get('steps',0)<=row['horizon']:
            raise ValueError('Invalid reference episode horizon/steps')
        actions=np.asarray(row.get('actions'),dtype=float)
        if actions.shape!=(row['steps'],7) or not np.isfinite(actions).all() or np.abs(actions).max()>1.00001:
            raise ValueError('Incomplete, nonfinite or unbounded executed action evidence')
    return dict(variant=variant,task=task,seen=task in TRAIN_SETS[variant],**summarize(rows))


def collect_matrix():
    cells=[];provenance={}
    for variant in TRAIN_SETS:
        _,_,metadata=verified_final(variant)
        output=WORK/variant/'eval15'
        config=json.loads((output/'config.json').read_text())
        complete=json.loads((output/'complete.json').read_text())
        expected=dict(variant=variant,train_tasks=TRAIN_SETS[variant],tasks=list(TASKS),episodes=15,
            protocol='reference',seed_start=10000,settling_steps=10,open_loop_steps=12,image_size=256,
            precision='fp32',checkpoint_sha256=metadata['checkpoint_sha256'],stats_sha256=metadata['stats_sha256'],
            primary_weights='ema_model_state_dict',all_selected_tasks_seen_by_release=False)
        if config!=expected or complete!={'total_episodes':75,'config':expected}:
            raise ValueError(f'Incomplete or different rollout contract: {variant}')
        hashes={}
        for task in TASKS:
            path=output/task/'episodes.json'
            cells.append(summarize_cell(json.loads(path.read_text()),variant,task))
            hashes[task]=sha256(path)
        provenance[variant]=dict(config=config,episode_sha256=hashes)
    return cells,provenance


def plot_matrix(cells,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(c['variant'],c['task']):c for c in cells}
    if set(lookup)!={(v,t) for v in TRAIN_SETS for t in TASKS} or len(cells)!=15:
        raise ValueError('Full unique3×5 matrix required')
    values=np.array([[lookup[v,t]['success_rate']*100 for t in TASKS] for v in TRAIN_SETS])
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(5),list(TASKS));ax.set_yticks(range(3),['Train D0','Train D0+D1','Train D0+D1+D2'])
    ax.set_xlabel('Evaluation task');ax.set_ylabel('TurboVLA training set')
    ax.set_title('TurboVLA generalization •15 episodes per cell',pad=14)
    for i,v in enumerate(TRAIN_SETS):
        for j,t in enumerate(TASKS):
            c=lookup[v,t]
            ax.text(j,i,f"{c['success_rate']:.0%}\n{c['successes']}/15 ·{'seen' if c['seen'] else 'unseen'}",
                    ha='center',va='center',fontsize=11,color='white' if values[i,j]>55 else '#202020')
    fig.colorbar(im,ax=ax,label='Success rate (%)',shrink=.85)
    fig.savefig(destination,dpi=180)
    plt.close(fig)


def report():
    cells,provenance=collect_matrix()
    output=ROOT/'docs'/'turbovla_multitask';output.mkdir(parents=True,exist_ok=True)
    write_json(output/'results.json',dict(cells=cells,provenance=provenance))
    with (output/'generalization_matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','train_set','task','seen','successes','episodes','success_rate','ci95_low','ci95_high'])
        writer.writeheader()
        for c in cells:
            writer.writerow({k:c[k] for k in ['variant','task','seen','successes','episodes','success_rate']}|
                            dict(train_set='+'.join(TRAIN_SETS[c['variant']]),ci95_low=c['ci95'][0],ci95_high=c['ci95'][1]))
    plot_matrix(cells,output/'generalization_matrix.png')
    lines=['# TurboVLA nested-task results','',
           'Three independent3001-update runs from original DINOv3/BERT/GroundingDINO; final EMA weights.',
           'Reference-protocol evaluation on seeds10000–10014,15 episodes per task.',
           'Modified no-noops data:46/47/48 demos; this differs from pi0’s50 native demos.',
           'This experiment uses batch8 and3001 updates, rather than the paper’s batch128 and40k updates.','',
           '| Train set | Task | Seen | Successes/N | SR | Wilson95% CI |',
           '|---|---|---|---|---|---|']
    for c in cells:
        low,high=c['ci95'];lines.append(f"| {'+'.join(TRAIN_SETS[c['variant']])} | {c['task']} | {c['seen']} | {c['successes']}/15 | {c['success_rate']:.1%} | {low:.1%}–{high:.1%} |")
    lines+=['','![Generalization matrix](generalization_matrix.png)','','H1/H2 never enter training or normalization. All seeds are shared across models.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('VERIFIED_MATRIX_WRITTEN',output,flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()
