"""Reference-protocol rollouts for the NEW trained checkpoints only."""
import argparse
import json
import sys
from pathlib import Path
from data import ROOT, WORK, TASKS, TRAIN_SETS
from prepare import write_json
from artifacts import verified_final
sys.path.insert(0,str(ROOT/'turbovla_eval'))
from eval_utils import configure, make_env, run_episode, summarize


def load_policy(variant, precision='fp32'):
    checkpoint,stats,metadata = verified_final(variant)
    configure()
    from turbovla.evaluation.suite_policy import TurboVLAPolicy
    return TurboVLAPolicy(ckpt_path=str(checkpoint),dinov3_path=str(WORK/'assets'/'dinov3'),
        bert_path=str(WORK/'assets'/'bert'),stats_path=str(stats),stats_key='train',
        precision=precision,allow_hf_download=False),metadata


def evaluate(variant, output=None, episodes=15, tasks=None, save_videos=True):
    tasks = list(TASKS) if tasks is None else tasks
    if not 1 <= episodes <= 50 or not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError('Require unique registered tasks and 1..50 episodes')
    output = Path(output) if output is not None else WORK/variant/'eval15'
    if output.exists():
        raise FileExistsError(f'Preserve prior rollout evidence: choose a new output: {output}')
    policy,metadata = load_policy(variant)
    output.mkdir(parents=True)
    config = dict(variant=variant,train_tasks=TRAIN_SETS[variant],tasks=tasks,episodes=episodes,
        protocol='reference',seed_start=10000,settling_steps=10,open_loop_steps=12,
        image_size=256,precision='fp32',checkpoint_sha256=metadata['checkpoint_sha256'],
        stats_sha256=metadata['stats_sha256'],primary_weights=metadata['primary_weights'],
        all_selected_tasks_seen_by_release=False)
    write_json(output/'config.json',config)
    summaries = {}
    for task_key in tasks:
        env,task,task_id,suite,horizon = make_env(task_key,'reference')
        rows = []
        try:
            for episode in range(episodes):
                video = output/'videos'/task_key/f'episode_{episode:03d}.mp4' if save_videos else None
                row = run_episode(env,policy,task.language,10000+episode,horizon,video_path=video)
                row.update(task=task_key,task_id=task_id,episode=episode,variant=variant,
                           seen_during_training=task_key in TRAIN_SETS[variant])
                rows.append(row)
                write_json(output/task_key/'episodes.json',rows)
                print(variant,task_key,episode+1,episodes,'success=',row['success'],'steps=',row['steps'],flush=True)
        finally:
            env.close()
        summaries[task_key] = summarize(rows)
        write_json(output/task_key/'summary.json',summaries[task_key])
        write_json(output/'summary.json',summaries)
    write_json(output/'complete.json',dict(total_episodes=episodes*len(tasks),config=config))
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=list(TRAIN_SETS),required=True)
    parser.add_argument('--episodes',type=int,default=15)
    parser.add_argument('--tasks',default='D0,D1,D2,H1,H2')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-videos',action='store_true')
    args = parser.parse_args()
    evaluate(args.variant,args.output,args.episodes,args.tasks.split(','),not args.no_videos)
