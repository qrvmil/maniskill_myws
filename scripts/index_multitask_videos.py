#!/usr/bin/env python3
"""Validate every final clip, require visible-only review notes, and build the video index."""
import argparse
import json
from pathlib import Path
import numpy as np
from maniskill_myws.pld.multitask_data import WORK,ROOT
from maniskill_myws.pld.multitask_protocol import TRAIN_SETS,TASKS,SEEDS,select_videos,validate_video_metadata,video_name
from maniskill_myws.pld.libero_sanity_videos import validate_video
from maniskill_myws.pld.libero_artifacts import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contact-sheets',action='store_true');a=parser.parse_args()
    notes_path=ROOT/'docs/multitask_sft/video_notes.json'
    notes=json.loads(notes_path.read_text()) if notes_path.exists() else {}
    clips=ROOT/'videos/multitask_sft';verified=[];missing=[]
    lines=['# Multi-task SFT rollout videos','',
        'Final 3,001-update checkpoints. Clips are the first two successes and first two failures in seed order, where available. Both policy camera views and the terminal observation are shown at 20 fps. Outcome-stratified clips illustrate behavior; full N=50 samples determine rates.','',
        '| Train set | Eval task | Seen? | Seed | Outcome | Video | Visible behavior note |',
        '|---|---|---|---:|---|---|---|']
    for variant,train in TRAIN_SETS.items():
        for task in TASKS:
            folder=WORK/'eval'/variant/'3001'/task
            if not (folder/'complete.json').exists():raise ValueError(f'Incomplete final cell: {variant}/{task}')
            rows=json.loads((folder/'episodes.json').read_text())
            if [r['seed'] for r in rows]!=list(SEEDS):raise ValueError('Final episodes incomplete')
            selected=select_videos(rows,2)
            manifest=json.loads((folder/'videos/manifest.json').read_text())
            if {m['seed'] for m in manifest}!={r['seed'] for r in selected}:raise ValueError('Wrong representative seeds')
            for row in selected:
                filename=video_name(variant,task,row);path=clips/filename
                meta=json.loads(path.with_suffix('.json').read_text())
                validate_video_metadata(meta,row,variant,task)
                decoded=validate_video(path,row['length']+1)
                if any(meta[k]!=decoded[k] for k in decoded):raise ValueError('Published video differs from recorded video')
                verified.append(dict(filename=filename,**decoded))
                if a.contact_sheets:
                    import imageio.v2 as imageio
                    from PIL import Image,ImageDraw
                    reader=imageio.get_reader(path)
                    indices=np.linspace(0,decoded['frames']-1,12,dtype=int)
                    sheet=Image.new('RGB',(decoded['width']*3,(decoded['height']+24)*4+40),'white')
                    draw=ImageDraw.Draw(sheet);draw.text((8,10),filename,fill='black')
                    for i,index in enumerate(indices):
                        frame=Image.fromarray(reader.get_data(int(index)))
                        x=(i%3)*decoded['width'];y=(i//3)*(decoded['height']+24)+40
                        sheet.paste(frame,(x,y));draw.text((x+4,y+decoded['height']+4),f'frame {index} / {decoded["frames"]-1}',fill='black')
                    reader.close();destination=WORK/'video_contact_sheets';destination.mkdir(exist_ok=True)
                    sheet.save(destination/(path.stem+'.png'))
                note=notes.get(filename)
                if not note:missing.append(filename);continue
                if not isinstance(note,str) or '|' in note:raise ValueError('Invalid visible behavior note')
                outcome='success' if row['success'] else 'failure'
                lines.append(f'| {"+".join(train)} | {task} | {"SEEN" if task in train else "HELD-OUT"} | {row["seed"]} | {outcome} | [{filename}](../videos/multitask_sft/{filename}) | {note} |')
            for success,label in [(True,'success'),(False,'failure')]:
                count=sum(bool(r['success'])==success for r in rows)
                if count==0:lines.append(f'| {"+".join(train)} | {task} | {"SEEN" if task in train else "HELD-OUT"} | — | No {label} episodes | — | 0/{len(rows)} in this category; no clip available. |')
    write_json(ROOT/'docs/multitask_sft/video_verification.json',verified)
    if missing:
        write_json(WORK/'video_notes_needed.json',missing)
        raise ValueError(f'{len(missing)} videos need visible-behavior notes; contact sheets are available if requested')
    (ROOT/'docs/MULTITASK_SFT_VIDEOS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
