"""Small, outcome-stratified rollout videos with independently decoded evidence."""
import json
from pathlib import Path
import subprocess

import numpy as np

from .libero_protocol import file_sha256


def select_video_rows(rows):
    counts={False:0,True:0}
    selected=[]
    for row in rows:
        outcome=bool(row['success'])
        if counts[outcome]<2:
            selected.append(row)
            counts[outcome]+=1
    return selected


def video_filename(model,task,seed,success):
    label=({'positive_control':'official_libero','lora_initialization':'lora_initialization'}.get(str(model))
           or f'step{int(model):04d}')
    return f'{label}_{task}_seed{seed}_{"success" if success else "failure"}.mp4'


def validate_replayed_row(reference,replay):
    for key in ('seed','success','length','reset_hash','trajectory_hash','image_hash'):
        if reference[key]!=replay[key]:
            raise ValueError(f'Video replay differs from recorded evaluation: {key}')


def validate_video(path,expected_frames):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError(f'Missing or empty video: {path}')
    decode=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-an','-f','null','-'],
                          capture_output=True,text=True)
    if decode.returncode:
        raise RuntimeError(f'Video does not decode fully: {path}: {decode.stderr}')
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                          '-show_entries','stream=nb_read_frames,width,height','-of','json',str(path)],
                         capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)['streams'][0]
    frames=int(stream['nb_read_frames'])
    if frames!=expected_frames:
        raise ValueError(f'Video frame count {frames} != expected {expected_frames}: {path}')
    return dict(frames=frames,width=int(stream['width']),height=int(stream['height']),
                bytes=path.stat().st_size,sha256=file_sha256(path),full_decode_passed=True)


def save_rollout_video(folder,model,task,row,transitions,*,origin):
    import imageio.v2 as imageio
    if not transitions or len(transitions)!=row['length']:
        raise ValueError('Rollout transitions do not match recorded episode length')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    name=video_filename(model,task,row['seed'],row['success'])
    path=folder/name
    if path.exists():raise FileExistsError(path)
    frames=[np.concatenate(t['images'],axis=1) for t in transitions]
    frames.append(np.concatenate(transitions[-1]['next_images'],axis=1))
    imageio.mimwrite(path,frames,fps=20)
    result=dict(model=str(model),task=task,filename=name,origin=origin,
                **{k:row[k] for k in ('seed','success','length','reset_hash','trajectory_hash','image_hash')},
                camera_views=['agentview','wrist'],fps=20,includes_terminal_frame=True,
                expected_frames=row['length']+1,
                **validate_video(path,row['length']+1))
    path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
