"""External compatible checkpoints must not acquire invented training labels."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_external_configuration_preserves_unknown_or_explicit_exposure():
    from maniskill_myws.pld.multitask_eval import EvalConfig
    unknown = EvalConfig('/foreign/checkpoint', 'external')
    assert unknown.is_seen('H1') is None
    known = EvalConfig('/foreign/checkpoint', 'external', seen_tasks=('D0', 'H1'))
    assert known.is_seen('H1') is True and known.is_seen('D2') is False
    assert EvalConfig('/primary', 'B').is_seen('D1') is True
    with pytest.raises(ValueError):
        EvalConfig('/primary', 'B', seen_tasks=('H1',))


def test_normalizer_discovery_is_unambiguous_and_primary_binding_stays_strict(tmp_path):
    from maniskill_myws.pld.multitask_eval import EvalConfig, infer_variant, resolve_normalization
    foreign = tmp_path / 'assets/foreign/norm_stats.json'
    foreign.parent.mkdir(parents=True); foreign.write_text('{}')
    assert infer_variant(tmp_path) == 'external'
    assert resolve_normalization(EvalConfig(str(tmp_path), 'external')) == foreign
    with pytest.raises(ValueError):
        resolve_normalization(EvalConfig(str(tmp_path), 'A'))
    primary = tmp_path / 'assets/local/multitask_A/norm_stats.json'
    primary.parent.mkdir(parents=True); primary.write_text('{}')
    assert infer_variant(tmp_path) == 'A'
    assert resolve_normalization(EvalConfig(str(tmp_path), 'A')) == primary
    with pytest.raises(ValueError):
        resolve_normalization(EvalConfig(str(tmp_path), 'external'))
    explicit = EvalConfig(str(tmp_path), 'external', normalization=str(foreign))
    assert resolve_normalization(explicit) == foreign


def test_external_multitask_cli_never_runs_the_primary_parallel_gate(tmp_path, monkeypatch):
    from maniskill_myws.pld.multitask_eval import build_parser
    from maniskill_myws.pld import multitask_parallel
    def forbidden():
        pytest.fail('An external checkpoint must not launch the primary validation gate')
    monkeypatch.setattr(multitask_parallel, 'validate_parallel', forbidden)
    args = build_parser().parse_args(['--checkpoint', str(tmp_path), '--task', 'all'])
    assert multitask_parallel.dispatch(args) is False


def test_external_rollout_and_video_keep_unknown_seen_status(tmp_path, monkeypatch):
    from maniskill_myws.pld.multitask_eval import EvalConfig, EvaluationSession, save_video
    from maniskill_myws.pld.multitask_protocol import prompt
    from maniskill_myws.pld import libero_runner
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    transitions = [dict(images=(frame, frame), next_images=(frame, frame))]
    row = dict(seed=10000, success=False, length=1, reset_hash='synthetic',
               trajectory_hash='synthetic', image_hash='synthetic', physics_states=[],
               task_progress=dict(minimum_reach_distance=.2))
    monkeypatch.setattr(libero_runner, 'run_episode', lambda *a, **kw: (dict(row), transitions))
    session = EvaluationSession.__new__(EvaluationSession)
    session.config = EvalConfig('/foreign/checkpoint', 'external')
    session.env = SimpleNamespace(prompt=prompt('H1')); session.task = 'H1'; session.base = None
    result, frames = session.run_one(10000)
    assert result['seen'] is None and result['variant'] == 'external'
    metadata = save_video(tmp_path, 'external', 'H1', result, frames)
    assert metadata['seen'] is None and metadata['train_tasks'] is None
    assert metadata['checkpoint'] == '/foreign/checkpoint'
    assert metadata['frames'] == 2 and (tmp_path / metadata['filename']).is_file()


def test_external_cli_forwards_exposure_and_uses_separate_output(tmp_path, monkeypatch):
    import sys
    from maniskill_myws.pld import multitask_eval, libero_artifacts, libero_runtime
    calls = {}
    class Artifacts:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Session:
        def __init__(self, config, **kwargs): calls['config'] = config
        def evaluate(self, task, output): calls.update(task=task, output=output)
        def close(self): calls['closed'] = True
    monkeypatch.setattr(libero_runtime, 'configure_base_inference', lambda config: None)
    monkeypatch.setattr(libero_artifacts, 'RunArtifacts', Artifacts)
    monkeypatch.setattr(multitask_eval, 'EvaluationSession', Session)
    monkeypatch.setattr(multitask_eval, 'WORK', tmp_path / 'work')
    checkpoint = tmp_path / 'foreign/42'
    monkeypatch.setattr(sys, 'argv', ['eval', '--checkpoint', str(checkpoint),
                                    '--seen-tasks', 'H1', '--task', 'H1', '--episodes', '2'])
    multitask_eval.main()
    assert calls['config'].variant == 'external' and calls['config'].is_seen('H1') is True
    assert calls['config'].seeds == (10000, 10001)
    assert calls['output'] == tmp_path / 'work/external_eval/42/H1'
    assert calls['closed']
