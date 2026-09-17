#!/usr/bin/env python3
"""Publish and independently verify selected videos, then require visual notes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_sanity_videos import select_video_rows,video_filename,validate_replayed_row,validate_video
from maniskill_myws.pld.libero_sanity import summarize_rows
from report_pi0_audit import MODELS,LABELS


def publish_videos(*, prepare=False):
    work=Path('/workspace/audit-run')
    output=ROOT/'docs/pi0_audit/videos';output.mkdir(parents=True,exist_ok=True)
    records=[];coverage=[]
    for model in MODELS:
        folder=work/'eval'/model
        if not (folder/'complete.json').exists():raise ValueError(f'Incomplete evaluation: {model}')
        for task in ('D0','D1'):
            rows=json.loads((folder/f'{task}_episodes.json').read_text())
            summarize_rows(rows)
            selected=select_video_rows(rows)
            for outcome in (True,False):
                available=sum(bool(r['success'])==outcome for r in rows)
                coverage.append(dict(model=model,task=task,outcome='success' if outcome else 'failure',
                                     available=available,required=min(2,available)))
            for row in selected:
                name=video_filename(model,task,row['seed'],row['success'])
                source=folder/'videos'/name
                metadata=json.loads(source.with_suffix('.json').read_text())
                if metadata['model']!=model or metadata['task']!=task or metadata['filename']!=name:
                    raise ValueError('Video identity mismatch')
                validate_replayed_row(row,metadata)
                if not metadata['includes_terminal_frame'] or metadata['expected_frames']!=row['length']+1:
                    raise ValueError('Video omits expected terminal observation')
                if metadata['origin']=='deterministic_replay':
                    replay_folder=folder/'video_replay'
                    if not (replay_folder/'complete.json').exists():raise ValueError('Replay not complete')
                    replay_rows=json.loads((replay_folder/f'{task}_episodes.json').read_text())
                    validate_replayed_row(row,next(r for r in replay_rows if r['seed']==row['seed']))
                elif metadata['origin']!='evaluation':raise ValueError('Unknown video origin')
                destination=output/name
                if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
                decoded=validate_video(destination,row['length']+1)
                if decoded['sha256']!=metadata['sha256']:raise ValueError('Video checksum changed')
                records.append(dict(metadata,**{'publication_path':str(destination.relative_to(ROOT)),
                                               'publication_verification':decoded}))
    expected={r['filename'] for r in records}
    if {p.name for p in output.glob('*.mp4')}!=expected:
        raise ValueError('Unexpected or missing MP4 in publication video directory')
    evidence=dict(videos=records,coverage=coverage,selection='First two episodes per observed outcome in seed order',
                  total_bytes=sum(r['bytes'] for r in records))
    (output.parent/'video_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if prepare:
        print(f'Prepared and fully decoded {len(records)} videos; visual notes still required.')
        return
    notes=json.loads((output.parent/'video_notes.json').read_text())
    if any(not notes.get(r['filename'],'').strip() for r in records):
        raise ValueError('Every video needs a human-inspected visible-behavior note')
    table='\n'.join(f"| {LABELS[r['model']]} | {r['task']} | {r['seed']} | {'Success' if r['success'] else 'Failure'} | [Video](pi0_audit/videos/{r['filename']}) | {notes[r['filename']]} |" for r in records)
    coverage_table='\n'.join(f"| {LABELS[c['model']]} | {c['task']} | {c['outcome'].capitalize()} | {c['available']}/50 | {c['required']} |" for c in coverage)
    doc=f'''# pi0 D0/D1 sanity audit: evaluation videos

These clips show the first two successes and first two failures in seed order for
each model and task, where available. They illustrate observed behavior; this
outcome-based sample does not estimate success rates. The full 50-episode results
per task remain in the [main report](PI0_D0_D1_SANITY_REPORT.md).
Notes describe sampled views spanning each clip, including its final observation;
they do not infer the policy's intent or hidden causes.

Each video shows **agentview on the left and wrist view on the right**, using the
same two cameras as the policy, downsampled to 128 pixels per view for recording.
Playback is 20 frames/second. A clip contains the pre-action observations and the
final post-action observation: exactly **episode length + 1 frames**.

The video requirement arrived after primary step 0 and the official control had
finished. Their selected episodes were replayed with identical seed, outcome,
length, reset hash, trajectory hash and image hash; original result rows were
preserved. The other clips were captured during their original evaluations.

All {len(records)} linked MP4s were checked after copying: each exists, is nonempty,
decodes fully, has the expected frame count, matches its recorded seed/outcome,
and matches its source checksum. Total video size: {evidence['total_bytes']/1024**2:.2f} MiB.
[Machine-readable verification](pi0_audit/video_verification.json).

| Model/checkpoint | Task | Seed | Outcome | Video path | Short note |
|---|---|---:|---|---|---|
{table}

## Outcome coverage

An outcome with zero available episodes has no example to save. Categories with
one episode retain that episode; categories with two or more retain two.

| Model/checkpoint | Task | Outcome | Available among 50 episodes | Videos saved |
|---|---|---|---:|---:|
{coverage_table}
'''
    (ROOT/'docs/PI0_D0_D1_SANITY_VIDEOS.md').write_text(doc)
    print(f'Indexed {len(records)} fully verified, visually reviewed videos.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true',help='Copy/verify evidence before human visual inspection')
    args=parser.parse_args()
    publish_videos(prepare=args.prepare)


if __name__=='__main__':main()
