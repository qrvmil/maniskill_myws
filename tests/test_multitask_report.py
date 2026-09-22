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
