"""Protocol checks and paired statistics. Missing provenance fails closed."""
import hashlib
import json
from pathlib import Path
import numpy as np


def task_key(task):
    return f"{task['suite']}/{task['name']}"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def residual_training_spec(config):
    """Regimen fields independent of task split and rollout observation contract."""
    spec = {key: config[key] for key in ('batch_size', 'buffer_capacity',
        'calql_updates', 'calql_n_actions', 'online_steps',
        'otf_backup_actions', 'otf_rollout_actions')}
    spec.update(warmup_episodes=config.get('warmup_episodes', 5),
                target_entropy=config.get('target_entropy'))
    for key in ('active_steps','warmup_actor_updates','visual_encoder','visual_encoder_sha256',
                'otf_include_base_action','otf_backup_entropy','updates_per_step','offline_fraction',
                'calql_alpha','calql_temp','calql_importance_sample','calql_max_target_backup','calql_backup_entropy',
                'probe_fraction','residual_actor_impl','residual_density','shared_visual_encoder','actor_q_reduction',
                'optimizer_warmup_steps','temperature_impl','weight_decay','calql_init',
                'residual_scale_start','residual_scale_warmup_steps','actor_update_interval','grad_clip_norm'):
        if key in config:spec[key]=config[key]
    return spec


def require_specialist_regimen(provenance, config, *, source_validation=False):
    if provenance.get('training_spec') != residual_training_spec(config):
        raise ValueError('Specialist training regimen differs from evaluation configuration')
    steps=provenance.get('training_steps',-1)
    max_steps=config['online_steps']
    if 'active_steps' in config:
        training_task=(config['residual_training_task'] if config.get('protocol_version',1)==2
                       else config['source'])
        horizon=training_task['horizon']
        max_steps=config.get('warmup_episodes',5)*horizon+config['active_steps']+horizon-1
        active=provenance.get('active_steps',-1)
        if not 0 <= active <= config['active_steps']+horizon-1 or active>steps:
            raise ValueError('Invalid recorded active training budget')
    valid_partial=source_validation and 0 <= steps <= max_steps
    if not valid_partial and steps != config['online_steps']:
        raise ValueError('Specialist has not completed the registered training budget')
    if not isinstance(provenance.get('sac_config'), dict):
        raise ValueError('Specialist is missing its actual SAC configuration')


class Protocol:
    def __init__(self, config):
        self.config = config
        self.protocol_version = config.get('protocol_version',1)
        if self.protocol_version == 1:
            self.base_alignment_task = config['source']
            self.residual_training_task = config['source']
            self.evaluation_tasks = config['tasks']
            self.base_alignment_key = task_key(self.base_alignment_task)
            self.residual_training_key = self.base_alignment_key
            self.source = self.residual_training_key  # Historical protocol alias.
            self.is_adaptation = False
            keys = [task_key(t) for t in self.evaluation_tasks]
            if len(keys) != len(set(keys)) or keys.count(self.source) != 1:
                raise ValueError('Tasks must be unique and include exactly one source')
            for task in self.evaluation_tasks:
                if (task_key(task) == self.source) != (task['distance'] == 'D0'):
                    raise ValueError('D0 must be exactly the training source')
            seed_names=('train_env_seeds','validation_env_seeds','eval_seeds')
            split_names=('source','tasks',*seed_names)
        elif self.protocol_version == 2:
            if 'source' in config or 'tasks' in config:
                raise ValueError('Protocol V2 uses explicit roles, not source/tasks aliases')
            self.base_alignment_task = config['base_alignment_task']
            self.residual_training_task = config['residual_training_task']
            self.evaluation_tasks = config['evaluation_tasks']
            self.base_alignment_key = task_key(self.base_alignment_task)
            self.residual_training_key = task_key(self.residual_training_task)
            self.is_adaptation = self.base_alignment_key != self.residual_training_key
            if self.base_alignment_task.get('distance') != 'D0':
                raise ValueError('Base alignment task must be D0')
            if self.residual_training_task.get('distance') != 'D1' or not self.is_adaptation:
                raise ValueError('Residual training task must be a distinct D1 task')
            evaluation_keys=[task_key(task) for task in self.evaluation_tasks]
            if (len(evaluation_keys)!=2 or len(set(evaluation_keys))!=2
                    or self.base_alignment_task not in self.evaluation_tasks
                    or self.residual_training_task not in self.evaluation_tasks):
                raise ValueError('Evaluation tasks must be exactly the two registered protocol roles')
            seed_names=('base_sanity_seeds','train_env_seeds','validation_env_seeds','eval_seeds')
            split_names=('protocol_version','base_alignment_task','residual_training_task',
                         'evaluation_tasks',*seed_names)
        else:
            raise ValueError(f'Unsupported protocol version: {self.protocol_version}')
        seed_blocks=[config[name] for name in seed_names]
        seed_sets=[set(seeds) for seeds in seed_blocks]
        if (any(not seeds or len(seeds)!=len(seed_set) for seeds,seed_set in zip(seed_blocks,seed_sets,strict=True))
                or any(seed_sets[i] & seed_sets[j] for i in range(len(seed_sets)) for j in range(i))):
            raise ValueError('Protocol seed blocks must be nonempty, duplicate-free and disjoint')
        self.split_hash = hashlib.sha256(json.dumps(
            {key:config[key] for key in split_names},sort_keys=True).encode()).hexdigest()
        execution={k:config[k] for k in ('replan_steps','render_size','rl_image_size','residual_scale')}
        if 'base_numerical_contract' in config:
            execution['base_numerical_contract']=config['base_numerical_contract']
        self.execution_hash = hashlib.sha256(json.dumps(execution,sort_keys=True).encode()).hexdigest()

    def require_training_task(self, task):
        matches=(task_key(task)==self.residual_training_key if self.protocol_version==1
                 else task==self.residual_training_task)
        if not matches:
            raise ValueError('Task does not match the registered residual-training task spec')

    def require_alignment_task(self, task):
        matches=(task_key(task)==self.base_alignment_key if self.protocol_version==1
                 else task==self.base_alignment_task)
        if not matches:
            raise ValueError('Task does not match the registered base-alignment task spec')

    def require_evaluation_task(self, task):
        if task not in self.evaluation_tasks:
            raise ValueError('Task does not match any registered evaluation task spec')

    def require_alignment(self, path):
        if not path:
            raise ValueError('A verified seen-only aligned-base manifest is required')
        manifest = json.loads(Path(path).read_text())
        if (manifest.get('training_tasks') != [self.base_alignment_key]
                or manifest.get('split_hash') != self.split_hash):
            raise ValueError('Alignment manifest task/split mismatch')
        if manifest.get('pretrained_checkpoint') != 'gs://openpi-assets/checkpoints/pi0_base':
            raise ValueError('Unapproved pretrained checkpoint; full-LIBERO alignment is forbidden')
        valid_steps=(manifest.get('alignment_steps')==3001 if self.protocol_version==2
                     else manifest.get('alignment_steps',0)>=1)
        if not valid_steps or not manifest.get('demonstrations'):
            raise ValueError('Manifest must record the registered SFT budget and demonstrations')
        if manifest.get('alignment_data_version')!='native_post_action_shift_v1':
            raise ValueError('Obsolete alignment timing contract; old engineering checkpoints cannot enter scientific runs')
        for entry in [*manifest['demonstrations'], manifest['normalization']]:
            if file_sha256(entry['path']) != entry['sha256']:
                raise ValueError('Alignment data/statistics checksum mismatch')
        if not Path(manifest['aligned_checkpoint']).exists():
            raise ValueError('Aligned checkpoint absent')
        verify_directory(manifest['aligned_checkpoint'],manifest.get('checkpoint_files',{}))
        return manifest


def paired_summary(base, residual):
    if not base or len(base) != len(residual):
        raise ValueError('Need nonempty equal episode counts')
    if len({r['seed'] for r in base}) != len(base):
        raise ValueError('Duplicate evaluation seeds')
    for a,b in zip(base,residual,strict=True):
        if (a['seed'],a['reset_hash']) != (b['seed'],b['reset_hash']):
            raise ValueError('Evaluation must pair identical seeds and reset states')
    a = np.array([r['success'] for r in base],float)
    b = np.array([r['success'] for r in residual],float)
    boot=np.random.default_rng(0).choice(b-a,size=(10000,len(a)),replace=True).mean(axis=1)
    return {'episodes':len(base), 'seeds':[r['seed'] for r in base],
            'base_successes':int(a.sum()), 'residual_successes':int(b.sum()),
            'SR_base':float(a.mean()), 'SR_residual':float(b.mean()),
            'delta_SR':float((b-a).mean()),
            'paired_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
            'residual_only_successes':int(np.sum((a==0)&(b==1))),
            'base_only_successes':int(np.sum((a==1)&(b==0))),
            'base_mean_length':float(np.mean([r['length'] for r in base])),
            'residual_mean_length':float(np.mean([r['length'] for r in residual]))}


def directory_manifest(root, *, exclude=()):
    root=Path(root)
    return {str(p.relative_to(root)):file_sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and str(p.relative_to(root)) not in exclude}


def verify_directory(root, expected, *, exclude=()):
    if not expected or directory_manifest(root,exclude=exclude)!=expected:
        raise ValueError('Directory contents differ from recorded provenance')


def assert_held_out(initial_state, demonstration_states):
    """Compare full qpos/qvel etc.; MuJoCo simulation time is not an initial condition."""
    state=np.asarray(initial_state,dtype=np.float64)
    demos=np.asarray(demonstration_states,dtype=np.float64)
    if demos.ndim!=2 or demos.shape[1:]!=state.shape:
        raise ValueError('Cannot verify holdout: simulator/demo state representations differ')
    if any(np.array_equal(state[1:],other[1:]) for other in demos):
        raise ValueError('Evaluation reset overlaps a source SFT demonstration initial state')


def require_zero_report(path, protocol, manifest_path):
    if not path:
        raise ValueError('A checkpoint-bound --zero-report is required before scientific rollout')
    report=json.loads(Path(path).read_text())
    expected_seeds=(protocol.config['base_sanity_seeds'] if protocol.protocol_version==2
                    else protocol.config['validation_env_seeds'])
    wrong_task=(protocol.protocol_version==2
                and report.get('task')!=protocol.base_alignment_key)
    if (report.get('passed') is not True or report.get('pairs',0)<2
            or report.get('alignment_sha256')!=file_sha256(manifest_path)
            or report.get('split_hash')!=protocol.split_hash
            or report.get('execution_hash')!=protocol.execution_hash
            or len(set(report.get('seeds',[])))!=report['pairs']
            or not set(report['seeds']).issubset(expected_seeds)
            or wrong_task):
        raise ValueError('Zero-residual equivalence report does not match this base/split')
    return report
