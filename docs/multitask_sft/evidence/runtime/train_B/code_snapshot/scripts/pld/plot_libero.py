#!/usr/bin/env python3
"""Export source-traceable static figures; never plot invented/pending gains."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maniskill_myws.pld.libero_protocol import file_sha256
from maniskill_myws.pld.libero_summary import gather_transfer_results

BLUE='#3268A8';INK='#23272B';GRAY='#B8BDC4';GRID='#E4E7EB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
                    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
                    'axes.spines.top':False,'axes.spines.right':False})


def save(fig,output,name):
    for suffix in ('png','svg'):
        fig.savefig(output/f'{name}.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)


def write_rows(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def sft(run,output):
    meta=json.loads((run/'metadata.json').read_text())
    manifest=json.loads((run/'alignment_manifest.json').read_text())
    if meta['status']!='COMPLETED' or len(manifest['training_tasks'])!=1:
        raise ValueError('SFT figure requires a completed source-only alignment run')
    path=run/'logs/sft_metrics.json'
    rows=json.loads(path.read_text()) if path.exists() else [json.loads(x) for x in (run/'logs/sft.jsonl').read_text().splitlines()]
    x=np.array([r['step'] for r in rows]);y=np.array([r['loss'] for r in rows])
    if len(rows)<20 or not np.isfinite(y).all() or not np.all(np.diff(x)==1):
        raise ValueError('Need at least20 consecutive finite observed updates; use a table otherwise')
    window=20
    protocol=json.loads((run/'protocol_config.json').read_text())
    median=np.array([np.median(y[i-window+1:i+1]) for i in range(window-1,len(y))])
    fig,ax=plt.subplots(figsize=(8,4.5));fig.subplots_adjust(top=.77,bottom=.21,left=.1,right=.97)
    fig.text(.1,.94,'Source-only pi0 SFT training loss',fontsize=15,weight='bold')
    fig.text(.1,.87,f'{len(rows):,} observed updates · batch {meta["batch_size"]} · one source task · training seed {protocol["training_seed"]}',fontsize=10)
    ax.scatter(x,y,s=9,color=GRAY,alpha=.65,label='Observed update loss')
    ax.plot(x[window-1:],median,color=BLUE,lw=1.8,label='Trailing 20-update median')
    ax.set(xlabel='Completed optimizer update',ylabel='Normalized flow-matching loss',ylim=(0,max(y)*1.08))
    ax.grid(axis='y',color=GRID,lw=.7);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=9)
    fig.text(.1,.055,'Training loss is not task success. Native trainer logs may be rounded.\nSource: '+str(run),fontsize=7.5,color=INK)
    save(fig,output,'sft_loss')
    for row in rows:row.update(source=manifest['training_tasks'][0],run=str(run))
    write_rows(output/'sft_input.csv',rows)
    return {'run':str(run),'metadata_sha256':file_sha256(run/'metadata.json'),
            'alignment_manifest_sha256':file_sha256(run/'alignment_manifest.json'),'updates':len(rows)}


def transfer(runs,output):
    result=gather_transfer_results(runs)
    write_rows(output/'tasks.csv',result['tasks']);write_rows(output/'buckets.csv',result['buckets'])
    groups=sorted({(r['source'],r['training_seed']) for r in result['buckets']})
    for index,(source,seed) in enumerate(groups):
        rows=sorted([r for r in result['buckets'] if (r['source'],r['training_seed'])==(source,seed)],key=lambda r:r['distance'])
        if len(rows)<2:raise ValueError('At least two observed distance buckets required; missing buckets are not zeros')
        values=np.array([100*r['mean_gain'] for r in rows]);positions=np.arange(len(rows))
        tasks=[r for r in result['tasks'] if (r['source'],r['training_seed'])==(source,seed)]
        counts=sorted({r['episodes'] for r in tasks})
        fig,ax=plt.subplots(figsize=(8,4.5));fig.subplots_adjust(top=.77,bottom=.23,left=.13,right=.97)
        fig.text(.13,.94,'Residual gain by ordinal task distance',fontsize=15,weight='bold')
        fig.text(.13,.87,f'Equal task-weight mean · training seed {seed} · paired episodes/task: {counts}',fontsize=10)
        for position,value in zip(positions,values,strict=True):
            ax.bar(position,value,width=.55,color=BLUE if value>=0 else 'white',edgecolor=BLUE,lw=1.5)
            ax.annotate(f'{value:+.1f}',(position,value),xytext=(0,5 if value>=0 else -5),
                        textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
        extent=max(5.,float(np.max(np.abs(values)))*1.3)
        ax.set(ylim=(-extent,extent),ylabel='SR residual − SR base (percentage points)',
               xticks=positions,xticklabels=[f'{r["distance"]}\n{r["tasks"]} task(s)' for r in rows])
        ax.axhline(0,color=INK,lw=1);ax.grid(axis='y',color=GRID,lw=.7);ax.set_axisbelow(True)
        fig.text(.13,.07,'Exploratory single-training-seed view; no training-seed confidence interval.\nMissing distance buckets are omitted. Source anchor: '+source.split('/')[-1],fontsize=7)
        save(fig,output,f'gain_distance_{index}')
    (output/'validated_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'runs':[str(p) for p in runs],'groups':len(groups),'validation':'Recomputed from completed paired episode records'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sft','transfer'])
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if args.mode=='sft' and len(args.runs)!=1:p.error('SFT chart takes exactly one completed run')
    source=sft(Path(args.runs[0]),output) if args.mode=='sft' else transfer(args.runs,output)
    source.update(command=shlex.join([sys.executable,*sys.argv]),
                  matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
                  plotter_sha256=file_sha256(__file__))
    shutil.copy2(__file__,output/'plotter_snapshot.py')
    (output/'chart_provenance.json').write_text(json.dumps(source,indent=2)+'\n')
    print(output)

if __name__=='__main__':main()
