"""Scientific invariants for the independent multi-task study."""
import copy
import numpy as np
import pytest


def module():
    from maniskill_myws.pld import multitask_protocol
    return multitask_protocol


def test_nested_tasks_and_heldout_leakage():
    m=module()
    assert list(m.TRAIN_SETS.values())==[('D0',),('D0','D1'),('D0','D1','D2')]
    assert len({(t['suite'],t['name']) for t in m.TASKS.values()})==5
    for v in m.TRAIN_SETS:
        m.require_training_tasks(v,m.TRAIN_SETS[v])
        with pytest.raises(ValueError):m.require_training_tasks(v,(*m.TRAIN_SETS[v],'H1'))
    assert len({m.prompt(t) for t in m.TASKS})==5


def test_balanced_sampler_ignores_unequal_task_lengths_and_is_reproducible():
    m=module()
    a=m.BalancedSampler([2,200,2000],24016,seed=0)
    indices=list(a)
    b=m.BalancedSampler([2,200,2000],24016,seed=0)
    assert indices==list(b)
    counts=np.bincount(np.searchsorted([2,202,2202],indices,side='right'),minlength=3)
    assert np.max(np.abs(counts/counts.sum()-1/3))<.015
    assert set(indices[:100]).intersection(range(2))=={0,1}
    with pytest.raises(ValueError):m.BalancedSampler([0,20],8)


def test_normalization_and_initialization_provenance(tmp_path):
    m=module();norm=tmp_path/'norm.json';norm.write_text('{}')
    p={'variant':'B','training_tasks':['D0','D1'],'demo_counts':{'D0':50,'D1':50},
       'normalization_path':str(norm),'normalization_sha256':m.sha256(norm),
       'normalization_training_tasks':['D0','D1'],'base_uri':m.BASE_URI,'base_manifest_sha256':'abc'}
    m.require_binding(p,'B','abc')
    for change in [{'normalization_training_tasks':['D0','H1']},{'base_manifest_sha256':'different'},
                   {'training_tasks':['D0']},{'demo_counts':{'D0':50,'D1':49}}]:
        with pytest.raises(ValueError):m.require_binding(dict(p,**change),'B','abc')
    norm.write_text('{"drift":1}')
    with pytest.raises(ValueError):m.require_binding(p,'B','abc')


def test_paired_statistics_reject_different_resets_and_duplicate_seeds():
    m=module()
    a=[dict(seed=i,reset_hash=str(i),success=i%2==0,length=20) for i in range(10)]
    b=copy.deepcopy(a);b[1]['success']=True;b[2]['success']=False
    result=m.paired_change(a,b)
    assert result['rescue']==result['harm']==1 and result['gain_pp']==0
    assert result['bootstrap_95_pp'][0]<0<result['bootstrap_95_pp'][1]
    b[0]['reset_hash']='wrong'
    with pytest.raises(ValueError):m.paired_change(a,b)
    with pytest.raises(ValueError):m.summarize(a+[a[0]])
    assert m.summarize(a)['success_rate']==.5


def test_video_selection_sorted_categories_and_metadata():
    m=module()
    rows=[dict(seed=i,success=i%2==0,length=4,reset_hash='r',trajectory_hash='t',image_hash='im') for i in range(7)]
    selected=m.select_videos(rows[::-1],2)
    assert [r['seed'] for r in selected]==[0,1,2,3]
    assert m.video_name('B','H1',selected[0])=='train_D0_D1_H1_seed0_success.mp4'
    meta=dict(selected[0],variant='B',task='H1',expected_frames=5,frames=5,includes_terminal_frame=True)
    m.validate_video_metadata(meta,selected[0],'B','H1')
    with pytest.raises(ValueError):m.validate_video_metadata(dict(meta,seed=3),selected[0],'B','H1')
    assert m.select_videos(rows,0)==[]


def test_image_sensitivity_measures_action_vectors():
    m=module();a=np.zeros((50,7));b=a.copy();b[:,0]=2
    result=m.action_changes(a,b)
    assert result=={'first_action_l2':2.,'first5_mean_l2':2.,'chunk_mean_l2':2.}
    with pytest.raises(ValueError):m.action_changes(a,b[:5])


def test_action_contract_and_prompt():
    m=module()
    from maniskill_myws.pld.libero_backend import ActionContract
    assert ActionContract().to_env(np.zeros(7)).shape==(7,)
    with pytest.raises(ValueError):ActionContract().to_env(np.zeros(8))
    assert m.prompt('H2')=='pick up the alphabet soup and place it in the basket'


def test_training_save_and_sampling_account_for_upstream_lookahead():
    from maniskill_myws.pld.multitask_train import sampling_summary,require_identical_initialization
    from maniskill_myws.pld.libero_sanity import completed_updates
    assert completed_updates(3000,3001)==3001
    # The official trainer fetches a lookahead batch after the final update.
    result=sampling_summary([0,1]*12,[2,2],updates=2,batch_size=8)
    assert result['optimizer_examples']==16 and sum(result['counts'])==16
    require_identical_initialization('same','same')
    with pytest.raises(ValueError):require_identical_initialization('a','b')


def test_conversion_rejects_heldout_before_opening_file(tmp_path):
    from maniskill_myws.pld.multitask_data import audit_h5
    pytest.importorskip('h5py')
    with pytest.raises(ValueError,match='Held-out'):audit_h5(tmp_path/'missing.h5','H1')


def test_evaluation_config_and_notebook_helper_contract():
    from maniskill_myws.pld.multitask_eval import EvalConfig,paired_reset,build_parser
    from types import SimpleNamespace
    cfg=EvalConfig(checkpoint='/tmp/checkpoint',variant='B',task='H1',seeds=(10000,10001),videos=2)
    assert cfg.seeds==(10000,10001)
    with pytest.raises(ValueError):EvalConfig(checkpoint='/tmp/c',variant='A',task='H1',seeds=(1,1))
    env=SimpleNamespace(reset=lambda **kw:('obs',dict(reset_hash='same')))
    registry={}
    assert paired_reset(env,10000,registry)[1]['reset_hash']=='same'
    env.reset=lambda **kw:('obs',dict(reset_hash='other'))
    with pytest.raises(ValueError):paired_reset(env,10000,registry)
    args=build_parser().parse_args(['--checkpoint','/tmp/c','--task','H1','--episodes','2','--seed-start','10000'])
    assert args.episodes==2 and args.task=='H1'


def test_cached_normalization_cannot_be_relabelled():
    from maniskill_myws.pld.multitask_data import validate_cached_provenance
    good={'variant':'B','training_tasks':['D0','D1'],'source_audit_sha256':'dataset',
          'normalization_sha256':'stats','recipe':'fixed'}
    validate_cached_provenance(good,good)
    with pytest.raises(ValueError):validate_cached_provenance(dict(good,source_audit_sha256='foreign'),good)
    with pytest.raises(ValueError):validate_cached_provenance(None,good)


def test_normalization_reads_exact_requested_file(tmp_path):
    pytest.importorskip('openpi')
    from maniskill_myws.pld.multitask_eval import load_normalization
    from openpi.shared.normalize import NormStats,serialize_json
    custom=tmp_path/'custom.json'
    custom.write_text(serialize_json({'state':NormStats(mean=np.array([3.]),std=np.array([2.]))}))
    (tmp_path/'norm_stats.json').write_text(serialize_json({'state':NormStats(mean=np.array([99.]),std=np.array([1.]))}))
    assert load_normalization(custom)['state'].mean[0]==3


def test_notebook_has_one_config_cell_and_reuses_evaluation_api():
    import json
    from pathlib import Path
    nb=json.loads(Path('notebooks/01_multitask_sft_eval.ipynb').read_text())
    parameters=[c for c in nb['cells'] if 'parameters' in c.get('metadata',{}).get('tags',[])]
    assert len(parameters)==1
    source=''.join(parameters[0]['source'])
    assert all(k in source for k in ('CHECKPOINT','TASK','N_EPISODES','SEED_LIST','VIDEOS_PER_OUTCOME'))
    codes='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'EvaluationSession(config)' in codes and 'session.evaluate(' in codes and 'session.run_one(' in codes
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')


def test_parallel_validation_requires_identical_full_rollouts():
    from maniskill_myws.pld.multitask_parallel import compare_rollouts
    reference=[dict(seed=10000,success=False,length=220,reset_hash='r',trajectory_hash='t',image_hash='i')]
    compare_rollouts(reference,copy.deepcopy(reference))
    for key in ('success','length','reset_hash','trajectory_hash','image_hash'):
        other=copy.deepcopy(reference);other[0][key]='different'
        with pytest.raises(ValueError):compare_rollouts(reference,other)


def test_parallel_source_drift_falls_back_to_serial(tmp_path,monkeypatch):
    import json
    from maniskill_myws.pld import multitask_parallel as m
    checkpoint=tmp_path/'A/checkpoints/pi0_libero_seen_lora32/EXP-001/0';checkpoint.mkdir(parents=True)
    validation=tmp_path/'parallel_validation';validation.mkdir()
    (validation/'decision.json').write_text(json.dumps({'safe':True,'source_signature':{'old':'source'}}))
    monkeypatch.setattr(m,'WORK',tmp_path)
    monkeypatch.setattr(m,'execution_signature',lambda:{'new':'source'},raising=False)
    assert m.validate_parallel() is False


def test_parallel_signature_covers_inference_and_config():
    from maniskill_myws.pld.multitask_parallel import execution_signature
    signature=execution_signature()
    assert 'libero_experiment.py' in signature['sources']
    assert 'multitask_data.py' in signature['sources']
    assert 'config' in signature and 'packages' in signature and 'gpu' in signature


def test_normalization_dimensions_reject_foreign_robot_contract():
    from types import SimpleNamespace
    from maniskill_myws.pld.multitask_eval import validate_normalization_contract
    stats={'state':SimpleNamespace(mean=np.zeros(8),std=np.ones(8)),
           'actions':SimpleNamespace(mean=np.zeros(7),std=np.ones(7))}
    validate_normalization_contract(stats)
    stats['actions']=SimpleNamespace(mean=np.zeros(8),std=np.ones(8))
    with pytest.raises(ValueError):validate_normalization_contract(stats)
