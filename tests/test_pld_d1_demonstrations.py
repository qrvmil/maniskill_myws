import importlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
from maniskill_myws.pld.libero_protocol import Protocol


@pytest.fixture
def api():
    try:
        return importlib.import_module('maniskill_myws.pld.libero_demonstrations')
    except ModuleNotFoundError:
        pytest.fail('Verified D1 demonstration producer is missing')


@pytest.fixture
def protocol():
    return Protocol(json.loads(Path('configs/pld_libero/d0_base_d1_residual.json').read_text()))


def test_shift_and_real_chunk_cache_alignment(api):
    class Model:
        def reset(self, seed):
            self.calls = []
            self.seed = seed
        def infer(self, obs):
            self.calls.append(obs['index'])
            return {'actions': np.repeat((np.arange(5) + obs['index'])[:, None] / 20, 7, axis=1)}
    model = Model()
    base = ChunkedBasePolicy(model, replan_steps=5)
    observations = [dict(index=i, state=np.full(8, i), images=np.full((2, 2, 2, 3), i, np.uint8)) for i in range(8)]
    actions = np.repeat((np.arange(8) / 10)[:, None], 7, axis=1)
    transitions, audit = api.build_demo_transitions(observations, actions, base, inference_seed=1234567, success_index=7)
    assert model.calls == [0, 5]
    assert model.seed == 1234567
    assert len(transitions) == 7
    for i, tr in enumerate(transitions):
        np.testing.assert_allclose(tr['action'], actions[i+1])
        np.testing.assert_allclose(tr['base_action'], i/20)
        np.testing.assert_allclose(tr['next_base_action'], (i+1)/20)
        assert not np.array_equal(tr['action'], tr['base_action'])
        assert tr['reward'] == float(i == 6)
        assert tr['done'] == (i == 6)
    assert audit['observation_count'] == 8
    assert audit['inference_seed'] == 1234567
    buffer = api.transitions_buffer(transitions)
    np.testing.assert_allclose(buffer.mc_returns[:7], .99 ** np.arange(6, -1, -1))
    # A new episode must reset both cache and inference RNG.
    api.build_demo_transitions(observations, actions, base, inference_seed=1234568, success_index=7)
    assert model.calls == [0, 5]
    assert model.seed == 1234568


def test_rejects_unsuccessful_and_initially_successful_demo(api):
    for index in (None, 0):
        with pytest.raises(ValueError, match='success'):
            api.build_demo_transitions([{}]*3, np.zeros((3, 7)), None, inference_seed=1000000, success_index=index)


def test_cross_task_h5_rejected(api, protocol, tmp_path):
    path = tmp_path/'wrong.hdf5'
    with h5py.File(path, 'w') as f:
        f.create_group('data').attrs['bddl_file_name'] = protocol.base_alignment_key+'.bddl'
    with pytest.raises(ValueError, match='D1|residual'):
        api.audit_d1_h5(path, protocol)


def test_reachability_component_and_whole_action_fractions(api):
    demo = np.zeros((2, 7)); demo[0, 0] = 1; demo[1, :] = .5
    result = api.reachability_report(demo, np.zeros((2, 7)), preclip_actions=np.full((2, 7), 1.2))
    assert result['component_representable_fraction'] == pytest.approx(13/14)
    assert result['whole_action_representable_fraction'] == .5
    assert result['mean_absolute_correction'] == pytest.approx(4.5/14)
    assert result['reconstruction_max_abs_error'] == .5
    assert result['projected_residual_saturation_fraction'] == pytest.approx(1/14)
    assert result['base_preclip_out_of_bounds_fraction'] == 1


def test_missing_and_forged_producer_evidence_rejected(api, protocol, tmp_path):
    manifest = tmp_path/'alignment.json'; manifest.write_text('{}')
    replay = tmp_path/'offline.npz'; replay.write_bytes(b'placeholder')
    for metadata in ({}, {'kind':'official_d1_demonstrations', 'verification_path':str(manifest),
                           'verification_sha256':'0'*64, 'inference_audit_path':str(manifest)}):
        with pytest.raises(ValueError, match='evidence|provenance'):
            api.validate_demonstration_evidence(replay, metadata, protocol, manifest)


def test_legacy_asset_relocation_preserves_scene(api, tmp_path):
    asset = tmp_path/'objects'/'bowl.msh'; asset.parent.mkdir(); asset.write_bytes(b'mesh')
    xml = '<mujoco><asset><mesh name="bowl" file="/Users/author/chiliocosm/assets/objects/bowl.msh"/></asset><worldbody><body pos="1 2 3"/></worldbody></mujoco>'
    relocated, evidence = api.relocate_legacy_assets(xml, tmp_path)
    assert str(asset) in relocated
    assert 'pos="1 2 3"' in relocated
    assert evidence[0]['sha256'] == api.file_sha256(asset)
    with pytest.raises(ValueError, match='asset'):
        api.relocate_legacy_assets(xml.replace('bowl.msh', 'missing.msh'), tmp_path)


@pytest.mark.parametrize('tamper', ['actions', 'base_actions', 'images', 'next_base_actions', 'verification'])
def test_rejects_payload_even_when_payload_sidecar_is_rehashed(api, protocol, tmp_path, tamper):
    """A payload digest alone must not replace source/chunk inference evidence."""
    from maniskill_myws.pld.libero_artifacts import write_json
    source = tmp_path/'demo.hdf5'
    with h5py.File(source, 'w') as f:
        data = f.create_group('data')
        data.attrs['bddl_file_name'] = api.D1_KEY+'.bddl'
        data.attrs['env_args'] = '{}'
        g = data.create_group('demo_0'); g.attrs['model_file'] = '<mujoco/>'
        g.create_dataset('actions', data=np.full((3,7), .2, np.float32))
        g.create_dataset('states', data=np.zeros((3,2)))
        obs = g.create_group('obs')
        for key, shape in [('ee_pos',(3,)),('ee_ori',(3,)),('gripper_states',(2,)),
                           ('agentview_rgb',(128,128,3)),('eye_in_hand_rgb',(128,128,3))]:
            obs.create_dataset(key, data=np.zeros((3,*shape), np.uint8 if 'rgb' in key else np.float32))
    class Model:
        def reset(self, seed): pass
        def infer(self, obs): return {'actions':np.zeros((5,7),np.float32)}
    with h5py.File(source, 'r') as f:
        observations = list(api.native_observations(f['data/demo_0'], 'test'))
        transitions, row = api.build_demo_transitions(observations, f['data/demo_0/actions'][:],
            ChunkedBasePolicy(Model()), inference_seed=1000000, success_index=2)
    buffer = api.transitions_buffer(transitions)
    manifest = tmp_path/'alignment.json'; manifest.write_text('{}')
    snapshot = tmp_path/'producer.py'; snapshot.write_text('test fixture only')
    verification = tmp_path/'verification.json'
    write_json(verification, dict(schema=api.SCHEMA, source=api.D1_KEY, source_h5=str(source),
        demo_h5_sha256=api.file_sha256(source), passed=True, rejected_demos=0,
        thresholds=dict(state_l2=.01,proprio_max_abs=.001,native_image_mae=5.),
        episodes=[dict(episode='demo_0',accepted=True,simulator_success=True,success_index=2,
            state_l2_max=0.,proprio_max_abs=0.,camera_mae_max_per_camera=[0.,0.],
            simulator_success_by_frame=[False,False,True])]))
    audit_path = tmp_path/'inference.json'
    metadata = dict(kind='official_d1_demonstrations', source=api.D1_KEY,
        base_alignment_task=protocol.base_alignment_key, base_alignment_key=protocol.base_alignment_key,
        residual_training_task=api.D1_KEY, residual_training_key=api.D1_KEY,
        split_hash=protocol.split_hash, execution_hash=protocol.execution_hash,
        alignment_sha256=api.file_sha256(manifest), gamma=.99,
        verification_path=str(verification), verification_sha256=api.file_sha256(verification),
        demo_h5_sha256=api.file_sha256(source), inference_audit_path=str(audit_path))
    audit = dict(schema=api.SCHEMA, source=api.D1_KEY, split_hash=protocol.split_hash,
        execution_hash=protocol.execution_hash, alignment_sha256=api.file_sha256(manifest),
        verification_sha256=api.file_sha256(verification), model_class='AlignedOpenPIModel', replan_steps=5,
        producer_snapshot_path=str(snapshot), producer_sha256=api.file_sha256(snapshot),
        episodes=[dict(episode='demo_0',**row,inference_calls=[dict(observation_index=0,
            preclip_actions=np.zeros((5,7)).tolist())])], replay_payload_sha256=api.replay_payload_sha256(buffer))
    write_json(audit_path,audit); metadata['inference_audit_sha256']=api.file_sha256(audit_path)
    replay=tmp_path/'offline.npz'; buffer.save(replay,**metadata)
    api.validate_demonstration_evidence(replay,metadata,protocol,manifest)
    if tamper == 'verification':
        altered=json.loads(verification.read_text()); altered['episodes'][0]['state_l2_max']=100.
        write_json(verification,altered)
        metadata['verification_sha256']=api.file_sha256(verification)
        audit['verification_sha256']=metadata['verification_sha256']
    else:
        getattr(buffer,tamper)[0].flat[0] = 1
    audit['replay_payload_sha256']=api.replay_payload_sha256(buffer)
    write_json(audit_path,audit); metadata['inference_audit_sha256']=api.file_sha256(audit_path)
    buffer.save(replay,**metadata)
    with pytest.raises(ValueError,match='evidence|source|cache'):
        api.validate_demonstration_evidence(replay,metadata,protocol,manifest)


def reconstructed_record():
    return dict(robot0_eef_pos=np.zeros((3,3),np.float32),
        robot0_eef_quat=np.tile([0.,0.,0.,1.],(3,1)),
        robot0_gripper_qpos=np.zeros((3,2),np.float32),
        agentview_image=np.zeros((3,256,256,3),np.uint8),
        robot0_eye_in_hand_image=np.zeros((3,256,256,3),np.uint8),
        physics=np.zeros((3,92)),executed_actions=np.full((3,7),.2,np.float32),
        success_flags=np.array([False,False,True]),initial_state=np.zeros(92))


def test_reexecution_pair_requires_matching_actual_successful_transitions(api):
    first=reconstructed_record(); second={k:v.copy() for k,v in first.items()}
    second['agentview_image'][0,0,0,0]=1
    result=api.verify_reexecution_pair(first,second)
    assert result['passed']
    assert result['success_index']==2
    assert result['repeat_image_max_abs']==1
    for key in ('physics','executed_actions','robot0_eef_pos','robot0_eye_in_hand_image','success_flags'):
        altered={k:v.copy() for k,v in first.items()}
        altered[key].flat[0]=2 if key != 'success_flags' else True
        with pytest.raises(ValueError,match='reexecution|success|determin'):
            api.verify_reexecution_pair(first,altered)
    failed={k:v.copy() for k,v in first.items()}; failed['success_flags'][:]=False
    with pytest.raises(ValueError,match='success'):
        api.verify_reexecution_pair(failed,failed)


def test_reexecuted_observations_use_actual_raw_state_and_runtime_adapter(api):
    record=reconstructed_record(); record['robot0_eef_pos'][1]=[1,2,3]
    record['agentview_image'][1]=17
    observations=list(api.reexecuted_observations(record,'D1'))
    np.testing.assert_equal(observations[1]['state'][:3],[1,2,3])
    assert observations[1]['images'].shape==(2,128,128,3)
    assert np.all(observations[1]['images'][0]==17)
    assert observations[1]['raw']['agentview_image'].shape==(256,256,3)


@pytest.mark.parametrize('tamper',['images','base_actions','inference_observation'])
def test_reexecuted_loader_binds_actual_records_and_inference(api,protocol,tmp_path,tamper):
    from maniskill_myws.pld.libero_artifacts import write_json
    source=tmp_path/'source.hdf5'; record=reconstructed_record()
    record['agentview_image'][:]=17  # Deliberately distinct from native source images.
    with h5py.File(source,'w') as f:
        data=f.create_group('data'); data.attrs['bddl_file_name']=api.D1_KEY+'.bddl'; data.attrs['env_args']='{}'
        g=data.create_group('demo_0'); g.attrs['model_file']='<mujoco/>'
        g.create_dataset('actions',data=record['executed_actions']); g.create_dataset('states',data=np.zeros((3,92)))
        obs=g.create_group('obs')
        for key,shape in [('ee_pos',(3,)),('ee_ori',(3,)),('gripper_states',(2,)),('agentview_rgb',(128,128,3)),('eye_in_hand_rgb',(128,128,3))]:
            obs.create_dataset(key,data=np.zeros((3,*shape),np.uint8 if 'rgb' in key else np.float32))
    path=tmp_path/'actual.npz'; np.savez_compressed(path,**record)
    pair=api.verify_reexecution_pair(record,record)
    verification=tmp_path/'verification.json'; native=tmp_path/'native.json'
    write_json(native,dict(passed=False,demo_h5_sha256=api.file_sha256(source)))
    write_json(verification,dict(schema=api.REEXECUTION_SCHEMA,source=api.D1_KEY,passed=True,
        source_h5=str(source),demo_h5_sha256=api.file_sha256(source),episodes=[dict(episode='demo_0',accepted=True,
        record_path=str(path),record_sha256=api.file_sha256(path),repeat_record_path=str(path),repeat_record_sha256=api.file_sha256(path),**pair)]))
    class Model:
        def reset(self,seed): pass
        def infer(self,raw): return {'actions':np.zeros((5,7),np.float32)}
    observations=list(api.reexecuted_observations(record,''))
    transitions,row=api.build_demo_transitions(observations,record['executed_actions'],ChunkedBasePolicy(Model()),inference_seed=1000000,success_index=2)
    buffer=api.transitions_buffer(transitions)
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    snapshot=tmp_path/'snapshot.py'; snapshot.write_text('synthetic test fixture only')
    audit_path=tmp_path/'inference.json'
    import hashlib
    raw_sha=hashlib.sha256(b''.join(np.asarray(observations[0]['raw'][key]).tobytes() for key in api.RAW_KEYS)).hexdigest()
    audit=dict(schema=api.REEXECUTION_SCHEMA,source=api.D1_KEY,split_hash=protocol.split_hash,
        execution_hash=protocol.execution_hash,alignment_sha256=api.file_sha256(manifest),
        verification_sha256=api.file_sha256(verification),model_class='AlignedOpenPIModel',replan_steps=5,
        producer_snapshot_path=str(snapshot),producer_sha256=api.file_sha256(snapshot),
        replay_payload_sha256=api.replay_payload_sha256(buffer),episodes=[dict(episode='demo_0',**row,
        inference_calls=[dict(observation_index=0,preclip_actions=np.zeros((5,7)).tolist(),raw_observation_sha256=raw_sha)])])
    write_json(audit_path,audit)
    metadata=dict(kind=api.REEXECUTION_KIND,source=api.D1_KEY,
        base_alignment_task=protocol.base_alignment_key,base_alignment_key=protocol.base_alignment_key,
        residual_training_task=api.D1_KEY,residual_training_key=api.D1_KEY,
        split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,
        alignment_sha256=api.file_sha256(manifest),gamma=.99,observation_source='current_runtime_action_reexecution_render256',
        source_native_alignment=False,demo_h5_sha256=api.file_sha256(source))
    for key,value in [('verification',verification),('native_verification',native),('inference_audit',audit_path)]:
        metadata[key+'_path']=str(value); metadata[key+'_sha256']=api.file_sha256(value)
    replay=tmp_path/'replay.npz'; buffer.save(replay,**metadata)
    api.validate_demonstration_evidence(replay,metadata,protocol,manifest)
    if tamper=='inference_observation': audit['episodes'][0]['inference_calls'][0]['raw_observation_sha256']='0'*64
    else: getattr(buffer,tamper)[0].flat[0]=99
    audit['replay_payload_sha256']=api.replay_payload_sha256(buffer)
    write_json(audit_path,audit); metadata['inference_audit_sha256']=api.file_sha256(audit_path)
    buffer.save(replay,**metadata)
    with pytest.raises(ValueError,match='evidence|inference'):
        api.validate_demonstration_evidence(replay,metadata,protocol,manifest)
