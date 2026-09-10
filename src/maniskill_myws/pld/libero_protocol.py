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
                'calql_alpha','calql_temp','calql_importance_sample','calql_max_target_backup','calql_backup_entropy'):
        if key in config:spec[key]=config[key]
    return spec


def require_specialist_regimen(provenance, config, *, source_validation=False):
    if provenance.get('training_spec') != residual_training_spec(config):
        raise ValueError('Specialist training regimen differs from evaluation configuration')
    steps=provenance.get('training_steps',-1)
    max_steps=config['online_steps']
    if 'active_steps' in config:
        horizon=config['source']['horizon']
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
        self.source = task_key(config['source'])
        keys = [task_key(t) for t in config['tasks']]
        if len(keys) != len(set(keys)) or keys.count(self.source) != 1:
            raise ValueError('Tasks must be unique and include exactly one source')
        for t in config['tasks']:
            if (task_key(t) == self.source) != (t['distance'] == 'D0'):
                raise ValueError('D0 must be exactly the training source')
        seed_sets = [set(config[k]) for k in ('train_env_seeds','validation_env_seeds','eval_seeds')]
        if any(not s for s in seed_sets) or any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Training, validation and evaluation seeds must be nonempty and disjoint')
        self.split_hash = hashlib.sha256(json.dumps({k:config[k] for k in (
            'source','tasks','train_env_seeds','validation_env_seeds','eval_seeds')},sort_keys=True).encode()).hexdigest()
        self.execution_hash = hashlib.sha256(json.dumps({k:config[k] for k in (
            'replan_steps','render_size','rl_image_size','residual_scale')},sort_keys=True).encode()).hexdigest()

    def require_training_task(self, task):
        if task_key(task) != self.source:
            raise ValueError('Unseen tasks are evaluation-only')

    def require_alignment(self, path):
        if not path:
            raise ValueError('A verified seen-only aligned-base manifest is required')
        manifest = json.loads(Path(path).read_text())
        if manifest.get('training_tasks') != [self.source] or manifest.get('split_hash') != self.split_hash:
            raise ValueError('Alignment manifest task/split mismatch')
        if manifest.get('pretrained_checkpoint') != 'gs://openpi-assets/checkpoints/pi0_base':
            raise ValueError('Unapproved pretrained checkpoint; full-LIBERO alignment is forbidden')
        if manifest.get('alignment_steps',0) < 1 or not manifest.get('demonstrations'):
            raise ValueError('Manifest must record actual SFT and demonstrations')
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
    if (report.get('passed') is not True or report.get('pairs',0)<2
            or report.get('alignment_sha256')!=file_sha256(manifest_path)
            or report.get('split_hash')!=protocol.split_hash
            or report.get('execution_hash')!=protocol.execution_hash
            or len(set(report.get('seeds',[])))!=report['pairs']
            or not set(report['seeds']).issubset(protocol.config['validation_env_seeds'])):
        raise ValueError('Zero-residual equivalence report does not match this base/split')
    return report
