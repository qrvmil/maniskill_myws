"""Task-aware summaries and publication figures from recorded episodes only."""
import csv
import json
from pathlib import Path
import numpy as np
from .multitask_protocol import TASKS,TRAIN_SETS,SEEDS,UPDATES,summarize,paired_change
from .libero_artifacts import write_json

LABELS={'A':'D0','B':'D0 + D1','C':'D0 + D1 + D2'}
COLORS={'A':'#245a81','B':'#c27c22','C':'#7a518a'}


def heldout_summary(tasks):
    rates={t:summarize(tasks[t])['success_rate'] for t in ('H1','H2')}
    rng=np.random.default_rng(0);boot=[]
    for t in ('H1','H2'):
        outcome=np.array([r['success'] for r in tasks[t]],float)
        boot.append(rng.choice(outcome,size=(10000,len(outcome))).mean(1))
    interval=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist()
    return dict(task_rates=rates,heldout_mean=float(np.mean(list(rates.values()))),
        stratified_bootstrap_95ci=interval,method='fixed tasks; equal task weights; seed bootstrap within each task; seed0,10000 draws')


def heldout_paired(before,after):
    paired={t:paired_change(before[t],after[t]) for t in ('H1','H2')}
    rng=np.random.default_rng(0);boot=[]
    for task in ('H1','H2'):
        a=sorted(before[task],key=lambda r:r['seed']);b=sorted(after[task],key=lambda r:r['seed'])
        delta=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
        boot.append(rng.choice(delta,size=(10000,len(delta))).mean(1))
    return dict(gain_pp=float(np.mean([p['gain_pp'] for p in paired.values()])),
        bootstrap_95_pp=(100*np.quantile(np.mean(boot,axis=0),[.025,.975])).tolist(),per_task=paired,
        method='equal-task mean of within-task paired seed bootstrap; fixed H1/H2; 10000 draws seed0')


def read_results(root,*,require_trajectory=True):
    root=Path(root);episodes={};summaries=[];sensitivity=[]
    for v in TRAIN_SETS:
        episodes[v]={}
        for update in UPDATES if require_trajectory else (3001,):
            episodes[v][update]={}
            for task in TASKS if update==3001 else ('D0','D1','D2'):
                folder=root/v/str(update)/task
                if not (folder/'complete.json').is_file():raise ValueError(f'Incomplete evaluation: {folder}')
                rows=json.loads((folder/'episodes.json').read_text())
                if [r['seed'] for r in rows]!=list(SEEDS):raise ValueError(f'Wrong final sample: {folder}')
                if any(r['variant']!=v or r['task']!=task or r['seen']!=(task in TRAIN_SETS[v]) for r in rows):
                    raise ValueError('Episode model/task/seen metadata differs')
                episodes[v][update][task]=rows
                summaries.append(dict(variant=v,train_set=LABELS[v],updates=update,task=task,
                    seen=task in TRAIN_SETS[v],**summarize(rows)))
        sensitivity.extend(json.loads((root/v/'3001/image_sensitivity.json').read_text()))
    # Every overlapping cell must share the exact task-specific resets.
    for task in TASKS:
        reference=episodes['A'][3001][task]
        for v in TRAIN_SETS:
            for update in episodes[v]:
                if task in episodes[v][update]:paired_change(reference,episodes[v][update][task])
    return episodes,summaries,sensitivity


def analysis(episodes):
    final={v:episodes[v][3001] for v in TRAIN_SETS}
    heldout={v:heldout_summary(final[v]) for v in TRAIN_SETS}
    comparison={}
    for a,b in [('A','B'),('A','C'),('B','C')]:
        comparison[a+'_to_'+b]=dict(tasks={t:paired_change(final[a][t],final[b][t]) for t in TASKS},
                                   heldout=heldout_paired(final[a],final[b]))
    seen_summary={}
    for v in TRAIN_SETS:
        seen=list(TRAIN_SETS[v]);unseen=[t for t in TASKS if t not in seen]
        rates={t:summarize(final[v][t])['success_rate'] for t in TASKS}
        seen_summary[v]=dict(seen_tasks=seen,unseen_tasks=unseen,
            mean_seen_sr=float(np.mean([rates[t] for t in seen])),
            mean_unseen_sr=float(np.mean([rates[t] for t in unseen])))
    return dict(heldout=heldout,comparisons=comparison,seen_unseen=seen_summary)


def figures(episodes,summaries,sensitivity,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#252525','text.color':'#252525',
        'savefig.dpi':220,'pdf.fonttype':42})
    final=[r for r in summaries if r['updates']==3001]
    lookup={(r['variant'],r['task']):r for r in final}
    def save(fig,name):
        fig.savefig(output/(name+'.png'),bbox_inches='tight')
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def axis(ax):
        ax.set_ylim(0,105);ax.set_ylabel('Success rate (%)');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    matrix=np.array([[100*lookup[v,t]['success_rate'] for t in TASKS] for v in TRAIN_SETS])
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(5),list(TASKS));ax.set_yticks(range(3),[LABELS[v] for v in TRAIN_SETS])
    ax.set_xlabel('Evaluation task');ax.set_ylabel('SFT train set')
    ax.set_title('Final task performance · 3,001 optimizer updates\nN = 50 paired resets per cell; outlined cells were seen in SFT',pad=15)
    for i,v in enumerate(TRAIN_SETS):
        for j,t in enumerate(TASKS):
            seen=t in TRAIN_SETS[v]
            ax.text(j,i,f'{matrix[i,j]:.0f}%\n{"SEEN" if seen else "HELD-OUT"}',ha='center',va='center',
                    color='white' if matrix[i,j]>55 else '#202020',fontsize=10)
            if seen:ax.add_patch(Rectangle((j-.46,i-.43),.92,.86,fill=False,edgecolor='#d5982d',lw=2.5))
    fig.colorbar(im,ax=ax,label='Success rate (%)',shrink=.88);save(fig,'generalization_matrix')
    heldout={v:heldout_summary(episodes[v][3001]) for v in TRAIN_SETS}
    fig,ax=plt.subplots(figsize=(7.4,4.8),layout='constrained')
    for task,color,marker in [('H1','#245a81','o'),('H2','#c27c22','s')]:
        y=[100*lookup[v,task]['success_rate'] for v in TRAIN_SETS]
        ci=np.array([lookup[v,task]['wilson_95ci'] for v in TRAIN_SETS])*100
        ax.errorbar([1,2,3],y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),label=task,
                    color=color,marker=marker,capsize=4,lw=1.8)
    ax.plot([1,2,3],[100*heldout[v]['heldout_mean'] for v in TRAIN_SETS],color='#252525',ls='--',marker='D',label='Equal-task mean',lw=2)
    axis(ax);ax.set_xticks([1,2,3]);ax.set_xlabel('Number of SFT training tasks')
    ax.set_title('Common held-out generalization\nH1/H2 excluded from every training corpus; task bars: Wilson 95% CI')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1));save(fig,'heldout_generalization')
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained');x=np.arange(5);width=.24
    for i,v in enumerate(TRAIN_SETS):
        y=np.array([lookup[v,t]['success_rate']*100 for t in TASKS]);ci=np.array([lookup[v,t]['wilson_95ci'] for t in TASKS])*100
        ax.bar(x+(i-1)*width,y,width,color=COLORS[v],label=LABELS[v],
            yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),capsize=3)
    axis(ax);ax.set_xticks(x,list(TASKS));ax.set_title('Per-task final performance · N = 50, Wilson 95% CI')
    ax.legend(title='SFT train set',frameon=False,ncol=3,loc='upper center');save(fig,'per_task_comparison')
    stage_names=['Reach <10 cm','Grasp contact','Lift >3 cm','Success']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True,layout='constrained')
    for ax,task in zip(axes,['D0','D1','H1']):
        for i,v in enumerate(TRAIN_SETS):
            rows=episodes[v][3001][task]
            values=[np.mean([r['ever_reached_10cm'] for r in rows]),
                np.mean([r['task_progress']['ever_grasped'] for r in rows]),
                np.mean([r['task_progress']['ever_lifted_3cm'] for r in rows]),np.mean([r['success'] for r in rows])]
            ax.bar(np.arange(4)+(i-1)*width,100*np.array(values),width,label=LABELS[v],color=COLORS[v])
        ax.set_title(task);ax.set_ylim(0,105);ax.set_xticks(range(4),['Reach','Contact','Lift','Success']);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[0].set_ylabel('Episodes meeting diagnostic (%)')
    axes[1].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.28))
    fig.supxlabel('Reach <10 cm · target-object grasp contact · lift >3 cm · binary task success (stages need not be nested)')
    save(fig,'behavior_stages')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True,layout='constrained')
    for ax,v in zip(axes,TRAIN_SETS):
        for t,color,marker in [('D0','#245a81','o'),('D1','#c27c22','s'),('D2','#7a518a','^')]:
            ys=[100*summarize(episodes[v][u][t])['success_rate'] for u in UPDATES]
            ax.plot(UPDATES,ys,label=t+(' (seen)' if t in TRAIN_SETS[v] else ' (held-out)'),
                color=color,marker=marker,ls='-' if t in TRAIN_SETS[v] else '--')
        axis(ax);ax.set_title('Train '+LABELS[v]);ax.set_xticks(UPDATES,rotation=40)
        ax.set_xlabel('Completed optimizer updates');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('SFT trajectory · N = 50 per point; update 0 is saved LoRA initialization')
    save(fig,'sft_trajectory')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained',sharey=True)
    metrics=['first_action_l2','first5_mean_l2','chunk_mean_l2']
    for ax,task in zip(axes,['D0','H1']):
        for i,v in enumerate(TRAIN_SETS):
            selected=[r for r in sensitivity if r['variant']==v and r['task']==task]
            if len(selected)!=10:raise ValueError('Incomplete image sensitivity bank')
            values=np.array([[r[m] for m in metrics] for r in selected])
            xpos=np.arange(3)+(i-1)*width
            ax.bar(xpos,values.mean(0),width,color=COLORS[v],label=LABELS[v],alpha=.8)
            for k in range(3):ax.scatter(np.repeat(xpos[k],len(values)),values[:,k],s=13,color=COLORS[v],edgecolor='white',linewidth=.4,zorder=3)
        ax.set_title(task);ax.set_xticks(range(3),['First action','First 5 mean','Full chunk mean'])
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[0].set_ylabel('Action change (L2, normalized OSC units)')
    axes[1].legend(frameon=False,title='SFT train set')
    fig.suptitle('Image shuffle sensitivity · 10 fixed observations/task\nBars: mean; dots: observations; state, prompt and flow noise held fixed')
    save(fig,'image_sensitivity')


def export(root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    episodes,summaries,sensitivity=read_results(root)
    stats=analysis(episodes)
    write_json(output/'statistics.json',stats);write_json(output/'summaries.json',summaries)
    write_json(output/'image_sensitivity.json',sensitivity)
    rows=[]
    for s in summaries:
        rows.append({k:v for k,v in s.items() if k!='wilson_95ci'}|dict(ci_low=s['wilson_95ci'][0],ci_high=s['wilson_95ci'][1]))
    with (output/'success_rates.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(episodes,summaries,sensitivity,output)
    return stats
