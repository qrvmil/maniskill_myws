"""Descriptive simulator diagnostics; these never modify the sparse reward."""
import numpy as np


def task_progress(env,raw):
    sim_env=env.env.env
    bowl=sim_env.sim.data.body_xpos[sim_env.obj_body_id['akita_black_bowl_1']]
    plate=sim_env.sim.data.body_xpos[sim_env.obj_body_id['plate_1']]
    gripper=np.asarray(raw['robot0_eef_pos'])
    return dict(reach_distance=float(np.linalg.norm(gripper-bowl)),
        reach_horizontal_distance=float(np.linalg.norm((gripper-bowl)[:2])),
        bowl_plate_distance=float(np.linalg.norm((bowl-plate)[:2])),
        bowl_height=float(bowl[2]),grasp=bool(sim_env._check_grasp(
            sim_env.robots[0].gripper,sim_env.objects_dict['akita_black_bowl_1'])))


def summarize_progress(rows):
    height=np.array([r['bowl_height'] for r in rows])
    distance=np.array([r['bowl_plate_distance'] for r in rows])
    return dict(minimum_reach_distance=min(r['reach_distance'] for r in rows),
        initial_reach_distance=rows[0]['reach_distance'],final_reach_distance=rows[-1]['reach_distance'],
        ever_grasped=any(r['grasp'] for r in rows),
        maximum_bowl_lift=float(height.max()-height[0]),
        ever_lifted_3cm=bool(height.max()-height[0]>.03),
        initial_bowl_plate_distance=float(distance[0]),final_bowl_plate_distance=float(distance[-1]),
        minimum_bowl_plate_distance=float(distance.min()),
        bowl_plate_progress=float(distance[0]-distance[-1]))
