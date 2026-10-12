import numpy as np
from eval_utils import ROOT, ASSETS, load_policy, make_env, reset_episode, evaluate

def main():
    policy=load_policy()
    env,task,*_=make_env('D0')
    try:
        observation,info=reset_episode(env,10000)
        keys=['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
        np.savez_compressed(ASSETS/'sample_observation.npz',**{k:observation[k] for k in keys})
        chunk=policy.predict_env_action_chunk_from_obs(observation,task.language,execute_steps=12)
        print('REAL CUDA FORWARD',chunk.shape,'finite=',np.isfinite(chunk).all(),'reset=',info,flush=True)
    finally: env.close()
    evaluate(policy,ROOT/'results/smoke',['D0','D1','D2','H1','H2'],episodes=1)

if __name__=="__main__": main()
