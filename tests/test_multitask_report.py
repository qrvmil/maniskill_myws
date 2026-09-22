import pytest


def test_heldout_mean_equal_weights_and_no_cross_task_episode_pooling():
    from maniskill_myws.pld.multitask_report import heldout_summary
    def rows(n,success):return [dict(seed=i,reset_hash=str(i),success=i<success,length=20) for i in range(n)]
    result=heldout_summary({'H1':rows(10,10),'H2':rows(50,0)})
    assert result['heldout_mean']==.5
    assert result['task_rates']=={'H1':1.,'H2':0.}


def test_equal_task_paired_aggregate_requires_pairing_within_each_task():
    from maniskill_myws.pld.multitask_report import heldout_paired
    a={'H1':[dict(seed=i,reset_hash=str(i),success=False,length=20) for i in range(10)],
       'H2':[dict(seed=i,reset_hash=str(i),success=True,length=20) for i in range(50)]}
    b={k:[dict(r) for r in rows] for k,rows in a.items()}
    for row in b['H1']:row['success']=True
    out=heldout_paired(a,b)
    assert out['gain_pp']==50
    b['H2'][0]['reset_hash']='wrong'
    with pytest.raises(ValueError):heldout_paired(a,b)


def test_report_rejects_wrong_checkpoint_weights_or_normalization():
    from maniskill_myws.pld.multitask_report import validate_evaluation_binding
    from maniskill_myws.pld.multitask_protocol import SEEDS
    rows=[dict(variant='B',task='H1',seen=False,seed=s,checkpoint='/checkpoints/3001') for s in SEEDS]
    binding=dict(variant='B',task='H1',seeds=list(SEEDS),checkpoint='/checkpoints/3001',
                 checkpoint_params={'params':'correct'},normalization_sha256='normalizer')
    saved=dict(optimizer_updates=3001,params_files={'params':'correct'},normalization_sha256='normalizer')
    validate_evaluation_binding(rows,binding,saved,'B','H1',3001)
    for changes in ({'checkpoint_params':{'params':'wrong'}},{'normalization_sha256':'foreign'},
                    {'checkpoint':'/checkpoints/2000'},{'variant':'C'}):
        with pytest.raises(ValueError):validate_evaluation_binding(rows,dict(binding,**changes),saved,'B','H1',3001)


def test_sensitivity_comparison_requires_same_observations_and_noise():
    from maniskill_myws.pld.multitask_report import validate_sensitivity
    from maniskill_myws.pld.multitask_protocol import SEEDS
    rows=[dict(variant=v,task=t,seed=s,donor_seed=SEEDS[(i+1)%10],bank_sha256=t,
               noise_sha256=str(s),first_action_l2=.1,first5_mean_l2=.2,chunk_mean_l2=.3)
          for v in 'ABC' for t in ('D0','H1') for i,s in enumerate(SEEDS[:10])]
    validate_sensitivity(rows)
    rows[-1]['noise_sha256']='different'
    with pytest.raises(ValueError):validate_sensitivity(rows)
    rows[-1]['noise_sha256']=str(rows[-1]['seed']);rows[-1]['bank_sha256']='different'
    with pytest.raises(ValueError):validate_sensitivity(rows)
