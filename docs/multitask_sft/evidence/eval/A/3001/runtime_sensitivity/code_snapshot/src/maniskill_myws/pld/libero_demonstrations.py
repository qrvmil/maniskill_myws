"""Fail-closed official D1 replay; never updates the frozen D0 model.

Native obs[i] is post-action[i], whereas states[i] is pre-action[i].
Replay starts at obs[0] and therefore uses action[1]. Native 128px images
pass through the runtime rotate180 adapter; online cameras render at 256px
before RL downsampling to 128px. This native-resolution difference is recorded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .libero_alignment import native_observation_action_indices
from .libero_backend import convert_observation
from .libero_protocol import file_sha256
from .libero_runner import insert_trajectory
from .replay_buffer import ReplayBuffer

D1_KEY = 'libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate'
SCHEMA = 'official_d1_verified_replay_v1'
INFERENCE_SEED_START = 1000000
STATE_L2_TOLERANCE = .01  # Official converter's divergence warning threshold.
PROPRIO_MAX_TOLERANCE = .001
IMAGE_MAE_TOLERANCE = 5.0  # Native uint8 render comparison, reported per camera.
ARRAY_FIELDS = ('obs', 'actions', 'base_actions', 'rewards', 'next_obs',
                'next_base_actions', 'dones', 'mc_returns', 'images', 'next_images')


def audit_d1_h5(path, protocol):
    import h5py
    if not protocol.is_adaptation or protocol.residual_training_key != D1_KEY:
        raise ValueError('Official D1 replay requires the registered D1 residual protocol')
    with h5py.File(path, 'r') as f:
        data = f['data']
        bddl = str(data.attrs.get('bddl_file_name', ''))
        if not bddl.endswith('/'+D1_KEY+'.bddl') and bddl != D1_KEY+'.bddl':
            raise ValueError('HDF5 must belong to the D1 residual task')
        content = data.attrs.get('bddl_file_content')
        if content is not None and hashlib.sha256(str(content).encode()).hexdigest() != protocol.residual_training_task['bddl_sha256']:
            raise ValueError('D1 HDF5 BDDL content differs from frozen task')
        names = sorted(data, key=lambda name: int(name.split('_')[-1]))
        if not names:
            raise ValueError('Empty D1 demonstrations')
        lengths = []
        for name in names:
            g = data[name]; a = g['actions'][:]; n = len(a)
            if n < 2 or a.shape != (n, 7) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
                raise ValueError(f'Invalid bounded controller actions: {name}')
            if len(g['states']) != n or not np.isfinite(g['states'][:]).all() or not g.attrs.get('model_file'):
                raise ValueError(f'Missing finite saved states/model XML: {name}')
            for key, shape in [('ee_pos', (3,)), ('ee_ori', (3,)), ('gripper_states', (2,)),
                               ('agentview_rgb', (128, 128, 3)), ('eye_in_hand_rgb', (128, 128, 3))]:
                if g['obs'][key].shape != (n, *shape):
                    raise ValueError(f'Invalid native observation shape: {name}/{key}')
            lengths.append(n)
        env_args = json.loads(data.attrs['env_args'])
    return dict(schema=SCHEMA, source=D1_KEY, path=str(Path(path).resolve()),
                sha256=file_sha256(path), episode_names=names, episode_lengths=lengths,
                env_args=env_args, bddl_content_present=content is not None,
                bddl_binding='exact recorded D1 task name; pinned runtime BDDL hash checked separately')


def native_observations(group, prompt, image_size=128):
    from robosuite.utils.transform_utils import axisangle2quat
    o = group['obs']
    for i in range(len(group['actions'])):
        raw = dict(robot0_eef_pos=o['ee_pos'][i], robot0_eef_quat=axisangle2quat(o['ee_ori'][i]),
                   robot0_gripper_qpos=o['gripper_states'][i], agentview_image=o['agentview_rgb'][i],
                   robot0_eye_in_hand_image=o['eye_in_hand_rgb'][i])
        yield dict(raw=raw, **convert_observation(raw, prompt, image_size=image_size))


def relocate_legacy_assets(xml, asset_root):
    """Relocate old chiliocosm asset prefixes only; retain original model XML."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    evidence = []
    for element in root.iter():
        old = element.get('file')
        if not old or '/chiliocosm/assets/' not in old:
            continue
        suffix = old.split('/chiliocosm/assets/', 1)[1]
        new = (Path(asset_root)/suffix).resolve()
        if not new.is_relative_to(Path(asset_root).resolve()) or not new.is_file():
            raise ValueError(f'Missing or foreign legacy asset: {old}')
        element.set('file', str(new))
        evidence.append(dict(original=old, resolved=str(new), sha256=file_sha256(new)))
    return ET.tostring(root, encoding='unicode'), evidence


def verify_demo(group, env):
    """Reproduce official converter from saved pre-action state and original XML.

    Never restore states between actions: doing so would conceal dynamics drift.
    Continue through the full recording to audit semantics, but replay terminates
    at the first genuine success after obs[0]. A first-frame success is rejected.
    """
    from libero.libero.utils.utils import postprocess_model_xml
    from libero.libero import get_libero_path
    from robosuite.utils.transform_utils import quat2axisangle
    env.reset()
    xml = postprocess_model_xml(str(group.attrs['model_file']), {})
    xml, relocated_assets = relocate_legacy_assets(xml, get_libero_path('assets'))
    env.reset_from_xml_string(xml)
    env.sim.reset()
    states = group['states'][:]
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    restored_state = env.sim.get_state().flatten().copy()
    actions = group['actions'][:]
    state_errors, proprio_errors, image_errors, successes = [], [], [], []
    replayed_states = []
    for i, action in enumerate(actions):
        raw, _, _, _ = env.step(action)
        replayed_states.append(env.sim.get_state().flatten().copy().tolist())
        if i+1 < len(states):
            state_errors.append(float(np.linalg.norm(env.sim.get_state().flatten()-states[i+1])))
        recorded = np.concatenate([group['obs'][k][i] for k in ('ee_pos', 'ee_ori', 'gripper_states')])
        actual = np.concatenate([raw['robot0_eef_pos'], quat2axisangle(raw['robot0_eef_quat']), raw['robot0_gripper_qpos']])
        proprio_errors.append(float(np.max(np.abs(recorded-actual))))
        image_errors.append([float(np.mean(np.abs(raw[r].astype(float)-group['obs'][s][i].astype(float))))
                             for r, s in [('agentview_image', 'agentview_rgb'),
                                          ('robot0_eye_in_hand_image', 'eye_in_hand_rgb')]])
        successes.append(bool(env._check_success()))
    first = next((i for i, success in enumerate(successes) if success), None)
    reasons = []
    if max(state_errors) > STATE_L2_TOLERANCE: reasons.append('saved_state_drift')
    if max(proprio_errors) > PROPRIO_MAX_TOLERANCE: reasons.append('post_action_proprioception_mismatch')
    if np.max(image_errors) > IMAGE_MAE_TOLERANCE: reasons.append('native_camera_mismatch')
    if first is None: reasons.append('no_simulator_success')
    elif first == 0: reasons.append('initial_observation_already_successful')
    return dict(accepted=not reasons, rejection_reasons=reasons, success_index=first,
                simulator_success=first is not None, simulator_success_by_frame=successes,
                state_l2_max=max(state_errors), state_l2_mean=float(np.mean(state_errors)),
                state_l2_by_step=state_errors, proprio_max_abs=max(proprio_errors),
                initial_saved_state=states[0].tolist(), restored_state=restored_state.tolist(),
                replayed_states=replayed_states,
                first_state_divergence=next((i for i,error in enumerate(state_errors) if error>STATE_L2_TOLERANCE),None),
                proprio_max_abs_by_step=proprio_errors,
                camera_mae_max_per_camera=np.max(image_errors, axis=0).tolist(),
                camera_mae_mean_per_camera=np.mean(image_errors, axis=0).tolist(),
                stored_terminal_reward=float(group['rewards'][-1]),
                stored_terminal_done=bool(group['dones'][-1]),
                original_model_xml_sha256=hashlib.sha256(str(group.attrs['model_file']).encode()).hexdigest(),
                processed_model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                relocated_assets=relocated_assets)


def build_demo_transitions(observations, actions, base, *, inference_seed, success_index):
    if success_index is None or not 0 < success_index < len(observations):
        raise ValueError('A verified simulator success after the initial observation is required')
    actions = np.asarray(actions, np.float32)
    if actions.shape != (len(observations), 7) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError('Invalid executed demonstration actions')
    base.reset(inference_seed)
    selected = observations[:success_index+1]
    # Each observation consumes exactly once, including the terminal observation.
    # Its next-base value is real and aligned even though the TD mask is zero.
    bases = [base.act(o.get('raw', o)) for o in selected]
    transitions = []
    for i, action_index in native_observation_action_indices(len(selected)):
        o, nxt = selected[i:i+2]
        terminal = action_index == success_index
        transitions.append(dict(obs=o['state'], action=actions[action_index], base_action=bases[i],
            reward=float(terminal), next_obs=nxt['state'], next_base_action=bases[action_index],
            done=terminal, images=o['images'], next_images=nxt['images']))
    return transitions, dict(inference_seed=int(inference_seed), observation_count=len(selected),
        transitions=len(transitions), base_clipped_components=base.clipped_components,
        base_actions_sha256=hashlib.sha256(np.asarray(bases, np.float32).tobytes()).hexdigest())


def transitions_buffer(transitions):
    if not transitions:
        raise ValueError('No verified successful transitions')
    buffer = ReplayBuffer(len(transitions), 8, 7, image_shape=transitions[0]['images'].shape)
    insert_trajectory(buffer, transitions, gamma=.99)
    return buffer


def reachability_report(actions, base_actions, *, preclip_actions):
    actions, base = np.asarray(actions), np.asarray(base_actions)
    preclip = np.asarray(preclip_actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape != base.shape or not len(actions):
        raise ValueError('Reachability requires paired nonempty seven-dimensional actions')
    if not all(np.isfinite(x).all() for x in (actions, base, preclip)) or np.any(np.abs(actions)>1) or np.any(np.abs(base)>1):
        raise ValueError('Reachability requires finite bounded executed/base actions')
    delta = actions-base
    absolute = np.abs(delta)
    unit_unclipped = delta/.5
    unit = np.clip(unit_unclipped, -1, 1)
    projected_preclip = base+.5*unit
    reconstruction = np.clip(projected_preclip, -1, 1)
    return dict(xi=.5, transitions=len(actions), mean_absolute_correction=float(absolute.mean()),
        per_dim_mean_absolute_correction=absolute.mean(axis=0).tolist(), quantile_levels=[.5,.9,.95,.99,1.],
        per_dim_absolute_quantiles=np.quantile(absolute,[.5,.9,.95,.99,1.],axis=0).tolist(),
        component_representable_fraction=float(np.mean(absolute<=.5)),
        per_dim_representable_fraction=np.mean(absolute<=.5,axis=0).tolist(),
        whole_action_representable_fraction=float(np.mean(np.all(absolute<=.5,axis=1))),
        base_preclip_out_of_bounds_fraction=float(np.mean(np.abs(preclip)>1)),
        base_preclip_max_abs=float(np.max(np.abs(preclip))),
        base_preclip_per_dim_mean_abs=np.mean(np.abs(preclip),axis=0).tolist(),
        projected_residual_saturation_fraction=float(np.mean(np.abs(unit_unclipped)>1)),
        projected_controller_clipping_fraction=float(np.mean(np.abs(projected_preclip)>1)),
        reconstruction_mean_abs_error=float(np.mean(np.abs(actions-reconstruction))),
        reconstruction_max_abs_error=float(np.max(np.abs(actions-reconstruction))))


def replay_payload_sha256(buffer):
    digest = hashlib.sha256()
    for name in ARRAY_FIELDS:
        array = np.ascontiguousarray(getattr(buffer, name)[:len(buffer)])
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def validate_demonstration_evidence(replay_path, metadata, protocol, manifest_path):
    """Loader gate: require hash-bound producer, simulator and inference evidence.

    This checks provenance consistency, not authenticity against a malicious
    party capable of rewriting every artifact and checksum.
    """
    if metadata.get('kind') == REEXECUTION_KIND:
        return validate_reexecuted_evidence(replay_path, metadata, protocol, manifest_path)
    expected = dict(kind='official_d1_demonstrations', source=protocol.residual_training_key,
        base_alignment_task=protocol.base_alignment_key, base_alignment_key=protocol.base_alignment_key,
        residual_training_task=protocol.residual_training_key, residual_training_key=protocol.residual_training_key,
        split_hash=protocol.split_hash, execution_hash=protocol.execution_hash,
        alignment_sha256=file_sha256(manifest_path), gamma=.99)
    if not protocol.is_adaptation or protocol.residual_training_key != D1_KEY or any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Demonstration evidence provenance mismatch')
    try:
        records = {}
        for kind in ('verification', 'inference_audit'):
            path = Path(metadata[kind+'_path'])
            if file_sha256(path) != metadata[kind+'_sha256']:
                raise ValueError(f'{kind} evidence checksum mismatch')
            records[kind] = json.loads(path.read_text())
        verification, audit = records['verification'], records['inference_audit']
        if verification['schema'] != SCHEMA or audit['schema'] != SCHEMA:
            raise ValueError('Unknown producer evidence schema')
        if verification['demo_h5_sha256'] != metadata['demo_h5_sha256'] or file_sha256(verification['source_h5']) != metadata['demo_h5_sha256']:
            raise ValueError('Source HDF5 evidence mismatch')
        if verification['source'] != D1_KEY or not verification['passed'] or verification['rejected_demos']:
            raise ValueError('Simulator verification evidence failed')
        if verification['thresholds'] != dict(state_l2=STATE_L2_TOLERANCE,
                proprio_max_abs=PROPRIO_MAX_TOLERANCE,native_image_mae=IMAGE_MAE_TOLERANCE):
            raise ValueError('Verification evidence thresholds differ from producer contract')
        for k in ('source', 'split_hash', 'execution_hash', 'alignment_sha256'):
            if audit[k] != expected[k]: raise ValueError('Inference evidence provenance mismatch')
        if (audit['model_class'] != 'AlignedOpenPIModel' or audit['replan_steps'] != 5
                or audit['verification_sha256'] != metadata['verification_sha256']
                or file_sha256(audit['producer_snapshot_path']) != audit['producer_sha256']):
            raise ValueError('Missing actual frozen model producer evidence')
        accepted = {r['episode']:r for r in verification['episodes'] if r['accepted']}
        if len(accepted) != len(audit['episodes']) or not accepted:
            raise ValueError('Verified episode evidence mismatch')
        seeds = []
        for row in audit['episodes']:
            verified = accepted[row['episode']]
            flags = verified['simulator_success_by_frame']
            first_success = next((i for i,success in enumerate(flags) if success),None)
            metrics = [verified['state_l2_max'],verified['proprio_max_abs'],
                       *verified['camera_mae_max_per_camera']]
            if (not np.isfinite(metrics).all() or verified['state_l2_max']>STATE_L2_TOLERANCE
                    or verified['proprio_max_abs']>PROPRIO_MAX_TOLERANCE
                    or max(verified['camera_mae_max_per_camera'])>IMAGE_MAE_TOLERANCE
                    or first_success != verified['success_index']):
                raise ValueError('Measured simulator verification evidence failed')
            if (not verified['simulator_success'] or verified['success_index'] != row['transitions']
                    or row['observation_count'] != row['transitions']+1
                    or row['inference_seed'] < INFERENCE_SEED_START):
                raise ValueError('Invalid inference/cache/terminal evidence')
            seeds.append(row['inference_seed'])
        if len(set(seeds)) != len(seeds): raise ValueError('Repeated inference seed evidence')
        with np.load(replay_path, allow_pickle=False) as data:
            size = int(data['size']); shape = tuple(data['image_shape'])
        buffer = ReplayBuffer(size,8,7,image_shape=shape); buffer.load(replay_path)
        if replay_payload_sha256(buffer) != audit['replay_payload_sha256']:
            raise ValueError('Replay payload differs from inference evidence')
        if sum(r['transitions'] for r in audit['episodes']) != size:
            raise ValueError('Replay length differs from episode evidence')
        offset = 0
        for row in audit['episodes']:
            n = row['transitions']; sl = slice(offset, offset+n)
            expected_terminal = np.zeros(n); expected_terminal[-1] = 1
            if (not np.array_equal(buffer.dones[sl],expected_terminal)
                    or not np.array_equal(buffer.rewards[sl],expected_terminal)
                    or not np.allclose(buffer.mc_returns[sl],.99**np.arange(n-1,-1,-1))):
                raise ValueError('Invalid sparse simulator terminal/MC evidence')
            # Bind every transition to the official source, not only an NPZ hash.
            import h5py
            with h5py.File(verification['source_h5'], 'r') as source:
                group = source['data'][row['episode']]
                if not np.array_equal(buffer.actions[sl],np.asarray(group['actions'][1:n+1],np.float32)):
                    raise ValueError('Replay actions differ from shifted source evidence')
                observations = list(native_observations(group, '', buffer.image_shape[1]))[:n+1]
                for i in range(n):
                    for key, values in [('obs',observations[i]['state']),('next_obs',observations[i+1]['state']),
                                        ('images',observations[i]['images']),('next_images',observations[i+1]['images'])]:
                        if not np.array_equal(getattr(buffer,key)[offset+i],values):
                            raise ValueError('Replay observations differ from native source evidence')
            calls = row['inference_calls']
            if [call['observation_index'] for call in calls] != list(range(0,n+1,5)):
                raise ValueError('Base cache inference observation evidence mismatch')
            chunks = np.asarray([call['preclip_actions'] for call in calls],np.float32)
            if chunks.shape != (len(calls),5,7) or not np.isfinite(chunks).all():
                raise ValueError('Invalid base preclip inference evidence')
            actual_bases = np.clip(chunks.reshape(-1,7)[:n+1],-1,1)
            if (not np.array_equal(buffer.base_actions[sl],actual_bases[:-1])
                    or not np.array_equal(buffer.next_base_actions[sl],actual_bases[1:])
                    or hashlib.sha256(actual_bases.tobytes()).hexdigest()!=row['base_actions_sha256']):
                raise ValueError('Replay base actions differ from actual inference/cache evidence')
            offset += n
        return audit
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Missing or malformed demonstration producer evidence') from exc


REEXECUTION_SCHEMA = 'official_d1_current_runtime_reexecution_v1'
REEXECUTION_KIND = 'official_d1_reexecuted_demonstrations'
RAW_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
            'agentview_image', 'robot0_eye_in_hand_image')


def reexecuted_observations(record, prompt, image_size=128):
    for i in range(len(record['executed_actions'])):
        raw = {key:record[key][i] for key in RAW_KEYS}
        yield dict(raw=raw, **convert_observation(raw, prompt, image_size=image_size))


def record_sha256(record, keys=None):
    digest = hashlib.sha256()
    for key in sorted(record if keys is None else keys):
        value = np.ascontiguousarray(record[key])
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def verify_reexecution_pair(first, second):
    """Compare two independently restored continuous action executions.

    Exact physics, actions, proprioception and success flags; raw camera max
    difference <=1 uint8, matching the existing online zero-residual contract.
    Native state/image agreement is intentionally a separate FAILED audit.
    """
    n = len(first['executed_actions'])
    if n < 2:
        raise ValueError('Reexecution needs success after the first observation')
    flags = np.asarray(first['success_flags'])
    if flags.shape != (n,) or not flags[-1] or flags[:-1].any():
        raise ValueError('Reexecution must stop at first actual simulator success')
    if first.keys() != second.keys():
        raise ValueError('Reexecution record schema mismatch')
    image_max = 0
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Reexecution shape/finite determinism check failed')
        if key in ('agentview_image', 'robot0_eye_in_hand_image'):
            if a.shape != (n,256,256,3) or a.dtype != np.uint8 or b.dtype != np.uint8:
                raise ValueError('Reexecution requires actual uint8 render256 cameras')
            image_max = max(image_max,int(np.max(np.abs(a.astype(np.int16)-b.astype(np.int16)))))
        elif not np.array_equal(a,b):
            raise ValueError(f'Reexecution deterministic {key} check failed')
    if image_max > 1:
        raise ValueError('Reexecution deterministic image check failed')
    if first['executed_actions'].shape != (n,7) or np.any(np.abs(first['executed_actions'])>1):
        raise ValueError('Reexecution actions violate bounded controller contract')
    return dict(passed=True, success_index=n-1, observation_count=n,
        repeat_physics_max_abs=0., repeat_proprio_max_abs=0., repeat_image_max_abs=image_max,
        first_record_sha256=record_sha256(first), second_record_sha256=record_sha256(second))


def record_reexecution(group, env, *, seed):
    """Original XML/state, actual current-runtime actions, no inter-step restores."""
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml
    env.seed(int(seed)); env.reset()
    xml = postprocess_model_xml(str(group.attrs['model_file']), {})
    xml, assets = relocate_legacy_assets(xml, get_libero_path('assets'))
    env.reset_from_xml_string(xml)
    sim = env.sim
    # Same MSAA-off setup as LiberoEnv, after rebuilding the original XML.
    sim.model.vis.quality.offsamples = 0
    context = sim._render_context_offscreen
    context.con.free(); context._set_mujoco_context_and_buffers()
    sim.reset(); sim.set_state_from_flattened(group['states'][0]); sim.forward()
    result = {key:[] for key in (*RAW_KEYS,'physics','executed_actions','success_flags')}
    initial = sim.get_state().flatten().copy()
    for action in group['actions'][:]:
        executed = np.asarray(action,np.float32)
        raw, _, _, _ = env.step(executed.tolist())
        for key in RAW_KEYS: result[key].append(np.asarray(raw[key]).copy())
        result['physics'].append(sim.get_state().flatten().copy())
        result['executed_actions'].append(executed)
        success = bool(env._check_success())
        result['success_flags'].append(success)
        if success: break
    result = {key:np.asarray(values) for key,values in result.items()}
    result['initial_state'] = initial
    n = min(len(result['physics']),len(group['states'])-1)
    drift = np.linalg.norm(result['physics'][:n]-group['states'][1:n+1],axis=1)
    diagnostic = dict(seed=int(seed), native_state_alignment=False,
        native_state_l2_max=float(np.max(drift)), native_state_l2_by_step=drift.tolist(),
        original_model_xml_sha256=hashlib.sha256(str(group.attrs['model_file']).encode()).hexdigest(),
        processed_model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), relocated_assets=assets)
    return result, diagnostic


def validate_reexecuted_evidence(replay_path, metadata, protocol, manifest_path):
    expected = dict(kind=REEXECUTION_KIND,source=D1_KEY,
        base_alignment_task=protocol.base_alignment_key,base_alignment_key=protocol.base_alignment_key,
        residual_training_task=D1_KEY,residual_training_key=D1_KEY,
        split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,
        alignment_sha256=file_sha256(manifest_path),gamma=.99,
        observation_source='current_runtime_action_reexecution_render256',source_native_alignment=False)
    if not protocol.is_adaptation or protocol.residual_training_key != D1_KEY or any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Reexecuted demonstration evidence provenance mismatch')
    try:
        reports = {}
        for key in ('verification','inference_audit','native_verification'):
            path = Path(metadata[key+'_path'])
            if file_sha256(path) != metadata[key+'_sha256']:
                raise ValueError('Reexecuted demonstration evidence checksum mismatch')
            reports[key] = json.loads(path.read_text())
        verification, audit = reports['verification'], reports['inference_audit']
        native = reports['native_verification']
        if (native.get('passed') is not False or native['demo_h5_sha256'] != metadata['demo_h5_sha256']
                or verification['schema'] != REEXECUTION_SCHEMA or audit['schema'] != REEXECUTION_SCHEMA
                or verification['source'] != D1_KEY or not verification['passed']):
            raise ValueError('Reexecuted/native contract evidence mismatch')
        source_path = verification['source_h5']
        if file_sha256(source_path) != metadata['demo_h5_sha256'] or verification['demo_h5_sha256'] != metadata['demo_h5_sha256']:
            raise ValueError('Reexecuted source evidence hash mismatch')
        audit_d1_h5(source_path,protocol)
        for key in ('source','split_hash','execution_hash','alignment_sha256'):
            if audit[key] != expected[key]: raise ValueError('Reexecution inference provenance mismatch')
        if (audit['model_class'] != 'AlignedOpenPIModel' or audit['replan_steps'] != 5
                or audit['verification_sha256'] != metadata['verification_sha256']
                or file_sha256(audit['producer_snapshot_path']) != audit['producer_sha256']):
            raise ValueError('Reexecuted frozen-base producer evidence mismatch')
        with np.load(replay_path,allow_pickle=False) as data:
            size=int(data['size']); image_shape=tuple(data['image_shape'])
        buffer=ReplayBuffer(size,8,7,image_shape=image_shape); buffer.load(replay_path)
        if not size or image_shape != (2,128,128,3) or replay_payload_sha256(buffer) != audit['replay_payload_sha256']:
            raise ValueError('Reexecuted replay payload evidence mismatch')
        verified = {row['episode']:row for row in verification['episodes'] if row['accepted']}
        if not verified or len(verified) != len(audit['episodes']):
            raise ValueError('Reexecuted episode evidence mismatch')
        import h5py
        offset=0; seen=set(); seeds=set()
        with h5py.File(source_path,'r') as source:
            for row in audit['episodes']:
                name=row['episode']; n=row['transitions']; sl=slice(offset,offset+n)
                if name in seen or row['inference_seed'] in seeds or row['inference_seed']<INFERENCE_SEED_START:
                    raise ValueError('Repeated reexecution episode/inference seed evidence')
                seen.add(name); seeds.add(row['inference_seed'])
                evidence=verified[name]
                records=[]
                for prefix in ('record','repeat_record'):
                    path=evidence[prefix+'_path']
                    if file_sha256(path) != evidence[prefix+'_sha256']:
                        raise ValueError('Reexecution actual trajectory evidence checksum mismatch')
                    with np.load(path,allow_pickle=False) as arrays:
                        records.append({k:arrays[k] for k in arrays.files})
                pair=verify_reexecution_pair(*records); record=records[0]
                if any(evidence[k] != v for k,v in pair.items()) or pair['success_index'] != n or row['observation_count'] != n+1:
                    raise ValueError('Reexecution repeated trajectory evidence mismatch')
                executed=np.asarray(source['data'][name]['actions'][:n+1],np.float32)
                if not np.array_equal(record['executed_actions'],executed) or not np.array_equal(buffer.actions[sl],executed[1:]):
                    raise ValueError('Reexecution action evidence differs from source')
                if not np.array_equal(record['initial_state'],source['data'][name]['states'][0]):
                    raise ValueError('Reexecution initial state differs from source evidence')
                observations=list(reexecuted_observations(record,''))
                for i in range(n):
                    for key,value in [('obs',observations[i]['state']),('next_obs',observations[i+1]['state']),
                                      ('images',observations[i]['images']),('next_images',observations[i+1]['images'])]:
                        if not np.array_equal(getattr(buffer,key)[offset+i],value):
                            raise ValueError('Replay differs from actual reexecution observation evidence')
                calls=row['inference_calls']
                if [call['observation_index'] for call in calls] != list(range(0,n+1,5)):
                    raise ValueError('Reexecuted base cache evidence mismatch')
                for call in calls:
                    raw=observations[call['observation_index']]['raw']
                    raw_hash=hashlib.sha256(b''.join(np.asarray(raw[key]).tobytes() for key in RAW_KEYS)).hexdigest()
                    if raw_hash!=call['raw_observation_sha256']:
                        raise ValueError('Actual base inference observation evidence mismatch')
                chunks=np.asarray([call['preclip_actions'] for call in calls],np.float32)
                if chunks.shape != (len(calls),5,7) or not np.isfinite(chunks).all():
                    raise ValueError('Reexecuted actual inference evidence malformed')
                bases=np.clip(chunks.reshape(-1,7)[:n+1],-1,1)
                if (not np.array_equal(buffer.base_actions[sl],bases[:-1])
                        or not np.array_equal(buffer.next_base_actions[sl],bases[1:])
                        or hashlib.sha256(bases.tobytes()).hexdigest()!=row['base_actions_sha256']):
                    raise ValueError('Reexecuted base actions differ from actual inference evidence')
                flags=np.zeros(n); flags[-1]=1
                if (not np.array_equal(buffer.dones[sl],flags) or not np.array_equal(buffer.rewards[sl],flags)
                        or not np.allclose(buffer.mc_returns[sl],.99**np.arange(n-1,-1,-1))):
                    raise ValueError('Reexecuted terminal/MC evidence mismatch')
                offset+=n
        if offset != size: raise ValueError('Reexecuted replay length evidence mismatch')
        return audit
    except (KeyError,OSError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError('Missing/malformed reexecuted demonstration evidence') from exc
