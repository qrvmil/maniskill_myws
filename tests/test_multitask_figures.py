"""Rendering smoke test uses explicitly synthetic rows only in pytest temporary paths."""
import numpy as np


def test_all_six_figures_render_png_and_pdf(tmp_path):
    from maniskill_myws.pld.multitask_protocol import TASKS,TRAIN_SETS,UPDATES,summarize
    from maniskill_myws.pld.multitask_report import figures
    episodes={};summaries=[];sensitivity=[]
    for v in TRAIN_SETS:
        episodes[v]={}
        for update in UPDATES:
            episodes[v][update]={}
            for task in TASKS if update==3001 else ('D0','D1','D2'):
                rows=[dict(seed=i,reset_hash=str(i),success=i%2==0,length=10,ever_reached_10cm=True,
                           task_progress=dict(ever_grasped=False,ever_lifted_3cm=False)) for i in range(50)]
                episodes[v][update][task]=rows
                summaries.append(dict(variant=v,task=task,updates=update,**summarize(rows)))
        for task in ('D0','H1'):
            for seed in range(10):sensitivity.append(dict(variant=v,task=task,first_action_l2=.1,first5_mean_l2=.2,chunk_mean_l2=.3))
    figures(episodes,summaries,sensitivity,tmp_path)
    assert len(list(tmp_path.glob('*.png')))==len(list(tmp_path.glob('*.pdf')))==6
    assert all(p.stat().st_size>1000 for p in tmp_path.iterdir())
