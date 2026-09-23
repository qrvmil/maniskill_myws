"""Reusable LIBERO evaluation and image-sensitivity API for compatible pi0 LoRA checkpoints."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .multitask_protocol import TASKS,TRAIN_SETS,SEEDS,prompt,sha256,summarize,select_videos,video_name,validate_video_metadata,action_changes
from .multitask_data import WORK,V4,train_config,repo_id
from .libero_artifacts import write_json
from .libero_protocol import directory_manifest


@dataclass
class EvalConfig:
    checkpoint: str
    variant: str
    task: str='H1'
    seeds: tuple=SEEDS
    videos: int=2
    normalization: str|None=None
    reset_dir: str|None=None

    def __post_init__(self):
        if self.variant not in TRAIN_SETS or self.task not in TASKS:raise ValueError('Unknown variant/task')
        if not self.seeds or len(set(self.seeds))!=len(self.seeds):raise ValueError('Seeds must be nonempty and unique')
        if self.videos<0:raise ValueError('Negative video count')
        self.seeds=tuple(sorted(self.seeds))


def paired_reset(env,seed,registry):
    raw,info=env.reset(seed=seed)
    key=str(seed)
    if key in registry and registry[key]!=info['reset_hash']:raise ValueError('Paired reset states differ')
    registry[key]=info['reset_hash']
    return raw,info


class PairedEnv:
    def __init__(self,env,registry,path):self._env=env;self.registry=registry;self.path=path
    def __getattr__(self,name):return getattr(self._env,name)
    def reset(self,*,seed):
        result=paired_reset(self._env,seed,self.registry)
        write_json(self.path,self.registry)
        return result


def load_normalization(path):
    from openpi.shared.normalize import deserialize_json
    return deserialize_json(Path(path).read_text())


def validate_normalization_contract(stats):
    for key,width in [('state',8),('actions',7)]:
        if key not in stats:raise ValueError('LIBERO normalization feature missing')
        mean=np.asarray(stats[key].mean);std=np.asarray(stats[key].std)
        if mean.shape!=(width,) or std.shape!=(width,) or not np.isfinite([mean,std]).all() or np.any(std<0):
            raise ValueError('Normalization does not match LIBERO 8D state / 7D OSC contract')


def save_video(folder,variant,task,row,transitions):
    import imageio.v2 as imageio
    from .libero_sanity_videos import validate_video
    if len(transitions)!=row['length'] or not transitions:raise ValueError('Video/rollout lengths differ')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/video_name(variant,task,row)
    if path.exists():raise FileExistsError(path)
    frames=[np.concatenate(t['images'],axis=1) for t in transitions]
    frames.append(np.concatenate(transitions[-1]['next_images'],axis=1))
    imageio.mimwrite(path,frames,fps=20)
    metadata=dict(variant=variant,train_tasks=list(TRAIN_SETS[variant]),task=task,
        seen=task in TRAIN_SETS[variant],filename=path.name,
        **{k:row[k] for k in ('seed','success','length','reset_hash','trajectory_hash','image_hash')},
        expected_frames=row['length']+1,includes_terminal_frame=True,fps=20,
        camera_views=['agentview','wrist'],origin='evaluation',**validate_video(path,row['length']+1))
    validate_video_metadata(metadata,row,variant,task)
    write_json(path.with_suffix('.json'),metadata);return metadata


class EvaluationSession:
    """Load once, then run_one/evaluate across tasks. All neural calls check prompt binding."""
    def __init__(self,config:EvalConfig,*,run=None):
        from .libero_runtime import configure_base_inference,require_base_inference_runtime
        configure_base_inference(V4);require_base_inference_runtime(V4)
        import torch
        from openpi.policies.policy_config import create_trained_policy
        from .libero_experiment import AlignedOpenPIModel
        from .libero_sanity import PromptCheckedPolicy
        from .libero_backend import ChunkedBasePolicy
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        self.config=config;cfg=train_config(config.variant)
        checkpoint=Path(config.checkpoint).resolve()
        norm=Path(config.normalization) if config.normalization else checkpoint/'assets'/repo_id(config.variant)/'norm_stats.json'
        if not norm.is_file():raise ValueError('Checkpoint normalization missing; supply normalization explicitly')
        self.binding=dict(checkpoint=str(checkpoint),variant=config.variant,normalization_path=str(norm),
            normalization_sha256=sha256(norm),action_horizon=cfg.model.action_horizon,replan_steps=5,
            action_contract='7D normalized LIBERO OSC, inverse norm once, clip [-1,1]',
            checkpoint_params=directory_manifest(checkpoint/'params'),config=repr(cfg))
        self.expected=[''];self.env=None
        norm_stats=load_normalization(norm)
        validate_normalization_contract(norm_stats)
        policy=create_trained_policy(cfg,checkpoint,norm_stats=norm_stats)
        self.checked=PromptCheckedPolicy(policy,lambda:self.expected[0])
        self.model=AlignedOpenPIModel.__new__(AlignedOpenPIModel)
        self.model.policy=self.checked;self.model.noise_shape=(cfg.model.action_horizon,cfg.model.action_dim)
        self.model.prompt='';self.model.inference_seconds=[]
        self.model.run=run or SimpleNamespace(begin_cuda_phase=lambda:None,end_cuda_phase=lambda _:None)
        self.base=ChunkedBasePolicy(self.model,replan_steps=V4['replan_steps'])

    def set_task(self,task,*,reset_dir=None):
        from .libero_backend import LiberoEnv
        if self.env is not None:self.env.close()
        env=LiberoEnv(TASKS[task],render_size=V4['render_size'])
        if env.prompt!=prompt(task):raise ValueError('Simulator prompt differs from registered task')
        self.task=task;self.model.prompt=self.expected[0]=env.prompt
        reset_dir=Path(reset_dir or self.config.reset_dir or WORK/'paired_resets');reset_dir.mkdir(parents=True,exist_ok=True)
        path=reset_dir/f'{task}.json'
        registry=json.loads(path.read_text()) if path.exists() else {}
        self.env=PairedEnv(env,registry,path)
        return env.prompt

    def run_one(self,seed):
        from .libero_runner import run_episode
        if self.env is None:self.set_task(self.config.task)
        stages=self.task in ('D0','D1','H1')
        row,transitions=run_episode(self.env,self.base,seed=seed,image_size=V4['rl_image_size'],record_progress=stages)
        row.pop('physics_states')
        row.update(variant=self.config.variant,task=self.task,prompt=self.env.prompt,
            seen=self.task in TRAIN_SETS[self.config.variant],checkpoint=self.config.checkpoint)
        if stages:row['ever_reached_10cm']=row['task_progress']['minimum_reach_distance']<.1
        return row,transitions

    def evaluate(self,task,output,*,seeds=None,videos=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/'complete.json').exists():raise FileExistsError(output/'complete.json')
        if (output/'episodes.json').exists():raise FileExistsError('Partial evaluation exists; preserve it and use another output')
        seeds=tuple(sorted(self.config.seeds if seeds is None else seeds))
        if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid episode seeds')
        count=self.config.videos if videos is None else videos
        self.set_task(task);rows=[];manifest=[]
        write_json(output/'binding.json',dict(self.binding,seeds=seeds,task=task))
        for seed in seeds:
            row,transitions=self.run_one(seed);rows.append(row)
            write_json(output/'episodes.json',rows)
            if seed in {r['seed'] for r in select_videos(rows,count)}:
                manifest.append(save_video(output/'videos',self.config.variant,task,row,transitions))
                write_json(output/'videos/manifest.json',manifest)
            print('MULTITASK_EVAL',self.config.variant,task,seed,row['success'],row['length'],flush=True)
        categories={label:dict(available=sum(bool(r['success'])==success for r in rows),
            saved=sum(bool(r['success'])==success for r in manifest)) for label,success in [('success',True),('failure',False)]}
        write_json(output/'video_categories.json',categories)
        write_json(output/'summary.json',summarize(rows))
        write_json(output/'prompt_evidence.json',self.checked.evidence)
        write_json(output/'complete.json',dict(variant=self.config.variant,task=task,n=len(rows),videos=len(manifest)))
        return rows

    def image_sensitivity(self,output,*,bank_dir=None):
        """10 initial observations/task; cyclic same-task image shuffle, identical flow noise."""
        from .libero_backend import openpi_observation
        bank_dir=Path(bank_dir or WORK/'image_bank');bank_dir.mkdir(parents=True,exist_ok=True)
        rows=[];correct_actions=[];shuffled_actions=[]
        for task in ('D0','H1'):
            self.set_task(task)
            path=bank_dir/f'{task}.npz'
            if not path.exists():
                observations=[];hashes=[]
                for seed in SEEDS[:10]:
                    raw,info=self.env.reset(seed=seed)
                    observations.append(openpi_observation(raw,prompt(task)));hashes.append(info['reset_hash'])
                np.savez_compressed(path,
                    state=np.stack([o['observation/state'] for o in observations]),
                    image=np.stack([o['observation/image'] for o in observations]),
                    wrist=np.stack([o['observation/wrist_image'] for o in observations]),
                    seeds=SEEDS[:10],reset_hashes=hashes,prompt=prompt(task))
            bank=np.load(path)
            if str(bank['prompt'])!=prompt(task) or not np.array_equal(bank['seeds'],SEEDS[:10]):
                raise ValueError('Image bank provenance mismatch')
            for i,seed in enumerate(bank['seeds']):
                j=(i+1)%len(bank['seeds'])
                obs={'observation/state':bank['state'][i],'observation/image':bank['image'][i],
                     'observation/wrist_image':bank['wrist'][i],'prompt':prompt(task)}
                shuffled=dict(obs,**{'observation/image':bank['image'][j],'observation/wrist_image':bank['wrist'][j]})
                noise=np.random.default_rng(int(seed)).standard_normal(self.model.noise_shape).astype(np.float32)
                correct=self.checked.infer(obs,noise=noise.copy())['actions']
                changed=self.checked.infer(shuffled,noise=noise.copy())['actions']
                correct_actions.append(np.array(correct,copy=True));shuffled_actions.append(np.array(changed,copy=True))
                rows.append(dict(variant=self.config.variant,task=task,seed=int(seed),
                    donor_seed=int(bank['seeds'][j]),bank_sha256=sha256(path),
                    noise_sha256=__import__('hashlib').sha256(noise.tobytes()).hexdigest(),
                    **action_changes(correct,changed)))
            print('IMAGE_SENSITIVITY_COMPLETE',self.config.variant,task,flush=True)
        action_path=Path(output).with_name('image_sensitivity_actions.npz')
        np.savez_compressed(action_path,correct=np.stack(correct_actions),shuffled=np.stack(shuffled_actions),
                            tasks=[r['task'] for r in rows],seeds=[r['seed'] for r in rows])
        write_json(output,rows);return rows

    def close(self):
        if self.env is not None:self.env.close();self.env=None


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--variant',choices=list(TRAIN_SETS),default=None)
    parser.add_argument('--task',default='H1',help='D0/D1/D2/H1/H2, comma-separated, or all')
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--seed-start',type=int,default=10000)
    parser.add_argument('--seeds',help='Explicit comma-separated seeds; overrides episode count')
    parser.add_argument('--videos',type=int,default=2,help='Per outcome and task')
    parser.add_argument('--normalization',default=None)
    parser.add_argument('--output',default=None)
    parser.add_argument('--image-sensitivity',action='store_true')
    parser.add_argument('--serial',action='store_true',help='Disable validated multi-process evaluation')
    parser.add_argument('--runtime-label',default='runtime')
    parser.add_argument('--reset-dir',default=None)
    parser.add_argument('--sensitivity-only',action='store_true')
    return parser


def main():
    import os
    from .libero_runtime import configure_base_inference
    from .libero_artifacts import RunArtifacts
    args=build_parser().parse_args()
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    configure_base_inference(V4)
    checkpoint=Path(args.checkpoint).resolve();variant=args.variant
    if variant is None:
        matches=[v for v in TRAIN_SETS if (checkpoint/'assets'/repo_id(v)/'norm_stats.json').exists()]
        if len(matches)!=1:raise ValueError('Cannot infer variant: supply --variant and --normalization')
        variant=matches[0]
    tasks=[] if args.sensitivity_only else (list(TASKS) if args.task=='all' else args.task.split(','))
    seeds=tuple(map(int,args.seeds.split(','))) if args.seeds else tuple(range(args.seed_start,args.seed_start+args.episodes))
    config=EvalConfig(str(checkpoint),variant,tasks[0] if tasks else 'D0',seeds,args.videos,args.normalization,args.reset_dir)
    output=Path(args.output) if args.output else WORK/'eval'/variant/checkpoint.name
    output.mkdir(parents=True,exist_ok=True)
    with RunArtifacts(output/args.runtime_label,vars(args)) as run:
        session=EvaluationSession(config,run=run)
        try:
            for task in tasks:session.evaluate(task,output/task)
            if args.image_sensitivity:session.image_sensitivity(output/'image_sensitivity.json')
        finally:session.close()
