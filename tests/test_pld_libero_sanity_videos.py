"""Video evidence must select outcomes consistently and verify real media."""
import shutil

import numpy as np
import pytest


def test_video_selection_keeps_first_two_per_outcome_and_rare_category():
    from maniskill_myws.pld.libero_sanity_videos import select_video_rows, video_filename
    rows=[dict(seed=9000+i,success=s) for i,s in enumerate([False,False,False,True,False,True,True])]
    assert [r['seed'] for r in select_video_rows(rows)]==[9000,9001,9003,9005]
    assert [r['seed'] for r in select_video_rows(rows[:4])]==[9000,9001,9003]
    assert video_filename('3001','D1',9017,False)=='step3001_D1_seed9017_failure.mp4'
    assert video_filename('positive_control','D0',9000,True)=='official_libero_D0_seed9000_success.mp4'


def test_replay_rejects_wrong_seed_outcome_or_trajectory():
    from maniskill_myws.pld.libero_sanity_videos import validate_replayed_row
    row=dict(seed=9000,success=False,length=220,reset_hash='r',trajectory_hash='t',image_hash='i')
    validate_replayed_row(row,dict(row))
    for key,value in [('seed',9001),('success',True),('length',219),('reset_hash','x'),('trajectory_hash','x'),('image_hash','x')]:
        with pytest.raises(ValueError):validate_replayed_row(row,dict(row,**{key:value}))


def test_saved_mp4_fully_decodes_and_has_exact_frame_count(tmp_path):
    pytest.importorskip('imageio')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg tools not installed')
    from maniskill_myws.pld.libero_sanity_videos import save_rollout_video, validate_video
    row=dict(seed=9000,success=False,length=3,reset_hash='r',trajectory_hash='t',image_hash='i')
    transitions=[{'images':np.full((2,32,32,3),i*60,np.uint8),
                  'next_images':np.full((2,32,32,3),(i+1)*60,np.uint8)} for i in range(3)]
    result=save_rollout_video(tmp_path,'0','D1',row,transitions,origin='evaluation')
    assert result['frames']==4 and result['width']==64 and result['height']==32
    assert result['seed']==9000 and result['success'] is False
    import imageio.v2 as imageio
    decoded=imageio.mimread(tmp_path/result['filename'])
    assert decoded[0].mean()<10 and decoded[-1].mean()>150
    with pytest.raises(ValueError):validate_video(tmp_path/result['filename'],3)
    (tmp_path/'broken.mp4').write_bytes(b'not a video')
    with pytest.raises((ValueError,RuntimeError)):validate_video(tmp_path/'broken.mp4',3)
