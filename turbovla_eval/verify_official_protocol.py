"""Real action-by-action parity against unchanged upstream rollout (isolated GPU run)."""
import json
import numpy as np
from eval_utils import ROOT,load_policy,evaluate,write_json,configure
configure()
from turbovla.evaluation.policy import set_seed_everywhere,get_libero_dummy_action,rotate_libero_image
from vla_adapter.rollout import GenerateConfig,_make_libero_env,_run_episode
from libero.libero import benchmark

if __name__=='__main__':
    policy=load_policy()
    output=ROOT/'results/official_parity'
    evaluate(policy,output,['D0'],episodes=2,protocol='official',save_videos=False)
    ours=json.loads((output/'D0/episodes.json').read_text())
    set_seed_everywhere(7)
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    task=suite.get_task(2)
    cfg=GenerateConfig(task_suite_name='libero_spatial',seed=7,mujoco_gl='egl',pyopengl_platform='egl')
    env,instruction=_make_libero_env(task,cfg)
    initial_states=suite.get_task_init_states(2)
    parity=[]
    try:
        for i in range(2):
            success,frames,actions=_run_episode(cfg,env,policy,instruction,initial_states[i],get_libero_dummy_action,rotate_libero_image)
            expected=np.asarray(ours[i]['actions']);actual=np.asarray(actions)
            assert expected.shape==actual.shape
            assert np.array_equal(expected,actual)
            assert success==ours[i]['success']
            parity.append(dict(episode=i,steps=len(actions),success=success,actions_identical=True))
    finally:env.close()
    write_json(output/'upstream_action_parity.json',parity)
    print(json.dumps(parity,indent=2))
