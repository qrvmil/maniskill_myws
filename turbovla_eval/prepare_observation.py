"""Create a real D0 observation in a separate simulator process before VRAM sampling."""
from eval_utils import ROOT, ASSETS, make_env, reset_episode, write_json, file_hash
import numpy as np

def prepare_observation():
    env,task,task_id,_,_=make_env('D0','reference')
    try:
        observation,info=reset_episode(env,10000)
        keys=['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
        path=ASSETS/'sample_observation.npz'
        np.savez_compressed(path,**{key:observation[key] for key in keys})
        write_json(path.with_suffix('.json'),dict(task='D0',instruction=task.language,task_id=task_id,
            seed=10000,protocol='reference',settle_steps=10,observation_sha256=file_hash(path),**info))
    finally:
        env.close()
        del env
        import gc
        gc.collect()

if __name__=='__main__': prepare_observation()
