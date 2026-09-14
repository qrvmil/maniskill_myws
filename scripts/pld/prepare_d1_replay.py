#!/usr/bin/env python3
"""Audit official native D1 trajectories and optionally infer frozen D0 actions."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np

from maniskill_myws.pld.libero_artifacts import RunArtifacts, write_json
from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
from maniskill_myws.pld.libero_demonstrations import (
    D1_KEY, SCHEMA, INFERENCE_SEED_START, STATE_L2_TOLERANCE, PROPRIO_MAX_TOLERANCE,
    IMAGE_MAE_TOLERANCE, audit_d1_h5, verify_demo, native_observations,
    build_demo_transitions, transitions_buffer, replay_payload_sha256,
    reachability_report, validate_demonstration_evidence, REEXECUTION_SCHEMA,
    REEXECUTION_KIND, record_reexecution, verify_reexecution_pair, reexecuted_observations)
from maniskill_myws.pld.libero_protocol import Protocol, file_sha256, require_zero_report


class InferenceRecorder:
    def __init__(self, model):
        self.model = model
        self.preclip = []
        self.calls = []

    def reset(self, seed):
        self.model.reset(seed)
        self.preclip = []
        self.calls = []

    def infer(self, raw):
        import hashlib
        result = self.model.infer(raw)
        actions = np.asarray(result['actions'], np.float32)[:5].copy()
        self.preclip.extend(actions)
        self.calls.append(dict(chunk_index=len(self.calls),
            observation_index=5*len(self.calls), preclip_actions=actions.tolist(),
            raw_observation_sha256=hashlib.sha256(b''.join(
                np.asarray(raw[k]).tobytes() for k in ('robot0_eef_pos', 'robot0_eef_quat',
                'robot0_gripper_qpos', 'agentview_image', 'robot0_eye_in_hand_image'))).hexdigest()))
        return result


def verify_source(source, protocol, run, max_demos=None):
    import h5py
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING
    audit = audit_d1_h5(source, protocol)
    write_json(run.path/'source_audit.json', audit)
    bddl = Path(get_libero_path('bddl_files'))/(D1_KEY+'.bddl')
    if file_sha256(bddl) != protocol.residual_training_task['bddl_sha256']:
        raise ValueError('Pinned D1 simulator BDDL differs from protocol')
    kwargs = dict(audit['env_args']['env_kwargs'])
    kwargs.update(bddl_file_name=str(bddl), hard_reset=False)
    env = TASK_MAPPING[audit['env_args']['problem_name']](**kwargs)
    report = dict(schema=SCHEMA, source=D1_KEY, source_h5=str(Path(source).resolve()),
        demo_h5_sha256=audit['sha256'], episodes=[], accepted_demos=0, rejected_demos=0, passed=False,
        thresholds=dict(state_l2=STATE_L2_TOLERANCE, proprio_max_abs=PROPRIO_MAX_TOLERANCE,
                        native_image_mae=IMAGE_MAE_TOLERANCE),
        semantics='states[i] pre-action[i]; obs[i] post-action[i]; obs[i] -> action[i+1] -> obs[i+1]',
        camera_contract='raw native128 -> runtime rotate180 -> RL128; online render256 -> rotate180 -> RL128; no reconstructed images',
        reward_contract='first actual simulator _check_success after obs[0]; source reward/done ignored',
        bddl_sha256=file_sha256(bddl), converter_sha256=file_sha256(Path(get_libero_path('bddl_files')).parents[2]/'scripts/create_dataset.py'))
    path = run.path/'verification.json'
    try:
        with h5py.File(source, 'r') as f:
            for name in audit['episode_names'][:max_demos]:
                try:
                    row = dict(episode=name, **verify_demo(f['data'][name], env))
                except Exception as exc:
                    row = dict(episode=name, accepted=False, rejection_reasons=['simulator_exception'], error=str(exc))
                report['episodes'].append(row)
                report['accepted_demos'] += int(row['accepted'])
                report['rejected_demos'] += int(not row['accepted'])
                write_json(path, report)
                print(json.dumps({k:row[k] for k in ('episode','accepted','rejection_reasons','success_index','state_l2_max','proprio_max_abs') if k in row}), flush=True)
                if not row['accepted']:
                    raise RuntimeError(f"D1 trajectory verification failed for {name}: {row['rejection_reasons']}")
        report['passed'] = report['accepted_demos'] > 0 and report['rejected_demos'] == 0
        report['complete_source'] = len(report['episodes']) == len(audit['episode_names'])
        write_json(path, report)
    finally:
        env.close()
    return report


def verify_reexecuted_source(source, protocol, run, native_path, max_demos=None):
    import h5py
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING
    audit = audit_d1_h5(source,protocol)
    native = json.loads(Path(native_path).read_text())
    if native.get('passed') is not False or native['demo_h5_sha256'] != audit['sha256']:
        raise ValueError('Reexecution must preserve the failed same-source native audit')
    write_json(run.path/'source_audit.json',audit)
    bddl=Path(get_libero_path('bddl_files'))/(D1_KEY+'.bddl')
    if file_sha256(bddl) != protocol.residual_training_task['bddl_sha256']:
        raise ValueError('Pinned reexecution D1 BDDL differs from protocol')
    kwargs=dict(audit['env_args']['env_kwargs'])
    kwargs.update(bddl_file_name=str(bddl),hard_reset=False,camera_heights=256,camera_widths=256)
    env=TASK_MAPPING[audit['env_args']['problem_name']](**kwargs)
    report=dict(schema=REEXECUTION_SCHEMA,source=D1_KEY,source_h5=str(Path(source).resolve()),
        demo_h5_sha256=audit['sha256'],native_verification_path=str(Path(native_path).resolve()),
        native_verification_sha256=file_sha256(native_path),episodes=[],accepted_demos=0,rejected_demos=0,
        passed=False,observation_source='current_runtime_action_reexecution_render256',
        source_native_alignment=False,bddl_sha256=file_sha256(bddl),
        semantics='actual post-action0 obs -> source action1 -> actual post-action1 obs; first actual success terminates',
        camera_contract='original XML cameras, render256, MSAA-off, existing rotate180 adapter -> RL128',
        deviation='current-runtime successful action reexecution; original native state/image correspondence failed and is not claimed',
        repeat_contract='independent XML+initial-state restores, exact physics/proprio/actions/success; raw uint8 image max<=1')
    (run.path/'reexecutions').mkdir()
    try:
        with h5py.File(source,'r') as f:
            for index,name in enumerate(audit['episode_names'][:max_demos]):
                group=f['data'][name]
                first,diagnostic=record_reexecution(group,env,seed=2000000+index)
                second,_=record_reexecution(group,env,seed=2000000+index)
                paths=[run.path/'reexecutions'/f'{name}.npz',run.path/'reexecutions'/f'{name}_repeat.npz']
                for path,record in zip(paths,(first,second),strict=True): np.savez_compressed(path,**record)
                row=dict(episode=name,**diagnostic,record_path=str(paths[0].resolve()),
                    record_sha256=file_sha256(paths[0]),repeat_record_path=str(paths[1].resolve()),
                    repeat_record_sha256=file_sha256(paths[1]))
                try:
                    pair=verify_reexecution_pair(first,second)
                    row.update(accepted=True,**pair)
                except ValueError as exc:
                    row.update(accepted=False,error=str(exc))
                report['episodes'].append(row)
                report['accepted_demos']+=int(row['accepted']); report['rejected_demos']+=int(not row['accepted'])
                write_json(run.path/'verification.json',report)
                print(json.dumps({k:row[k] for k in ('episode','accepted','success_index','repeat_image_max_abs','native_state_l2_max','error') if k in row}),flush=True)
                # Unsuccessful source actions are honest rejections, but any
                # nondeterminism is a broken evidence contract and stops the run.
                if not row['accepted'] and first['success_flags'].any():
                    raise RuntimeError(row['error'])
        report['passed']=report['accepted_demos']>0
        report['complete_source']=len(report['episodes'])==len(audit['episode_names'])
        write_json(run.path/'verification.json',report)
        if not report['passed']: raise RuntimeError('No actual successful D1 reexecutions')
    finally:
        env.close()
    return report


def produce(source, cfg, protocol, manifest, alignment_path, verification, run):
    import h5py
    from maniskill_myws.pld.libero_experiment import AlignedOpenPIModel
    import maniskill_myws.pld.libero_demonstrations as producer
    model = AlignedOpenPIModel(cfg, manifest, run)
    model.prompt = protocol.residual_training_task['name'].replace('_', ' ')
    recorder = InferenceRecorder(model)
    base = ChunkedBasePolicy(recorder, replan_steps=5)
    all_transitions, episodes, all_preclip = [], [], []
    reconstructed=verification['schema']==REEXECUTION_SCHEMA
    with h5py.File(source, 'r') as f:
        for index, verified in enumerate(verification['episodes']):
            if not verified['accepted']: continue
            name = verified['episode']; group = f['data'][name]
            if reconstructed:
                with np.load(verified['record_path'],allow_pickle=False) as data:
                    record={k:data[k] for k in data.files}
                observations=list(reexecuted_observations(record,model.prompt,cfg['rl_image_size']))
                actions=record['executed_actions']
            else:
                observations = list(native_observations(group, model.prompt, cfg['rl_image_size']))
                actions=group['actions'][:]
            transitions, audit = build_demo_transitions(observations, actions, base,
                inference_seed=INFERENCE_SEED_START+index, success_index=verified['success_index'])
            episodes.append(dict(episode=name, **audit, inference_calls=recorder.calls))
            all_transitions.extend(transitions)
            all_preclip.extend(recorder.preclip[:len(transitions)])
            write_json(run.path/'inference_progress.json', episodes)
    buffer = transitions_buffer(all_transitions)
    snapshot = run.path/'producer_snapshot.py'; shutil.copyfile(producer.__file__, snapshot)
    verification_path = run.path/'verification.json'
    audit_path = run.path/'inference_audit.json'
    evidence = dict(schema=verification['schema'], model_class='AlignedOpenPIModel', replan_steps=5,
        source=D1_KEY, split_hash=protocol.split_hash, execution_hash=protocol.execution_hash,
        alignment_sha256=file_sha256(alignment_path), verification_sha256=file_sha256(verification_path),
        producer_snapshot_path=str(snapshot.resolve()), producer_sha256=file_sha256(snapshot),
        inference_seed_rule='1000000 + numeric-order source episode index; distinct from all outcome reset seeds',
        episodes=episodes, replay_payload_sha256=replay_payload_sha256(buffer))
    write_json(audit_path, evidence)
    write_json(run.path/'reachability.json', reachability_report(buffer.actions, buffer.base_actions,
        preclip_actions=np.asarray(all_preclip)))
    metadata = dict(kind=REEXECUTION_KIND if reconstructed else 'official_d1_demonstrations', source=D1_KEY,
        base_alignment_task=protocol.base_alignment_key, base_alignment_key=protocol.base_alignment_key,
        residual_training_task=D1_KEY, residual_training_key=D1_KEY,
        split_hash=protocol.split_hash, execution_hash=protocol.execution_hash,
        alignment_sha256=file_sha256(alignment_path), gamma=.99,
        demo_h5_sha256=verification['demo_h5_sha256'],
        verification_path=str(verification_path.resolve()), verification_sha256=file_sha256(verification_path),
        inference_audit_path=str(audit_path.resolve()), inference_audit_sha256=file_sha256(audit_path))
    if reconstructed:
        metadata.update(observation_source='current_runtime_action_reexecution_render256',source_native_alignment=False,
            native_verification_path=verification['native_verification_path'],
            native_verification_sha256=verification['native_verification_sha256'])
    replay_path = run.path/'offline.npz'
    buffer.save(replay_path, **metadata)
    validate_demonstration_evidence(replay_path, metadata, protocol, alignment_path)
    write_json(run.path/'checksums.json', {p.name:file_sha256(p) for p in
        (replay_path, verification_path, audit_path, run.path/'reachability.json', snapshot)})
    run.meta.update(transitions=len(buffer), successes=len(episodes), checkpoint=manifest['aligned_checkpoint'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--alignment-manifest')
    parser.add_argument('--zero-report')
    parser.add_argument('--source-h5', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--reexecute',action='store_true',help='Distinct actual current-runtime action reexecution contract')
    parser.add_argument('--native-verification',help='Immutable failed native correspondence audit')
    parser.add_argument('--feasibility-report',help='Completed 100-reset D1 base collection report')
    parser.add_argument('--max-demos', type=int, help='Limit simulator-only audit; forbidden for replay production')
    args = parser.parse_args()
    if args.max_demos is not None and (not args.verify_only or args.max_demos < 1):
        parser.error('--max-demos requires --verify-only and a positive limit')
    cfg = json.loads(Path(args.config).read_text()); protocol = Protocol(cfg)
    if cfg['replan_steps'] != 5 or cfg['residual_scale'] != .5 or cfg['rl_image_size'] != 128:
        parser.error('D1 replay requires replan5, xi=.5, and RL128')
    manifest = None
    if args.reexecute and not args.native_verification:
        parser.error('--reexecute requires the preserved --native-verification audit')
    if not args.verify_only:
        if not args.alignment_manifest or not args.zero_report:
            parser.error('Production requires frozen --alignment-manifest and --zero-report')
        from maniskill_myws.pld.libero_runtime import configure_base_inference
        configure_base_inference(cfg)
        manifest = protocol.require_alignment(args.alignment_manifest)
        require_zero_report(args.zero_report, protocol, args.alignment_manifest)
        if args.reexecute:
            if not args.feasibility_report: parser.error('Reexecuted production requires --feasibility-report')
            feasibility=json.loads(Path(args.feasibility_report).read_text())
            rows=feasibility['episodes']
            if (feasibility['attempts']!=100 or len(rows)!=100 or feasibility['successes']>=50
                    or [r['seed'] for r in rows]!=cfg['train_env_seeds']
                    or sum(bool(r['success']) for r in rows)!=feasibility['successes']):
                raise ValueError('Fallback requires completed registered D1 feasibility with fewer than 50 successes')
    cfg['command_options'] = vars(args)
    with RunArtifacts(args.output, cfg) as run:
        shutil.copyfile(__file__, run.path/'prepare_d1_replay.py')
        verification = (verify_reexecuted_source(args.source_h5,protocol,run,args.native_verification,args.max_demos)
                        if args.reexecute else verify_source(args.source_h5, protocol, run, args.max_demos))
        run.meta.update(verify_only=args.verify_only, accepted_demos=verification['accepted_demos'])
        if not args.verify_only:
            produce(args.source_h5, cfg, protocol, manifest, args.alignment_manifest, verification, run)


if __name__ == '__main__':
    main()
