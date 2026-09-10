import json
from pathlib import Path
import pytest
from maniskill_myws.pld.libero_protocol import Protocol,file_sha256


def test_selection_rejects_final_unseen_or_incomplete_validation(tmp_path):
    from maniskill_myws.pld.libero_selection import select_source_validation
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    def evaluation(name,successes):
        p=tmp_path/name;(p/'eval').mkdir(parents=True)
        manifest=p/'alignment.json';manifest.write_text(json.dumps({'alignment_steps':int(name)}))
        (p/'config.json').write_text(json.dumps(dict(cfg,command_options=dict(alignment_manifest=str(manifest)))))
        (p/'metadata.json').write_text(json.dumps({'status':'COMPLETED'}))
        row=dict(source=Protocol(cfg).source,target=Protocol(cfg).source,distance='D0',
            evaluation_scope='source_validation',seeds=cfg['validation_env_seeds'],episodes=20,
            SR_base=successes/20,base_successes=successes,alignment_sha256=file_sha256(manifest))
        (p/'eval/summary.json').write_text(json.dumps([row]));return p,row
    a,_=evaluation('1000',5);b,row=evaluation('2000',8)
    result=select_source_validation([a,b],cfg,role='base')
    assert result['selected_evaluation']==str(b.resolve())
    assert result['success_rate']==.4
    for changes in [dict(evaluation_scope='held_out_evaluation'),dict(distance='D1'),dict(seeds=cfg['eval_seeds'][:20]),dict(episodes=10)]:
        (b/'eval/summary.json').write_text(json.dumps([dict(row,**changes)]))
        with pytest.raises(ValueError):select_source_validation([a,b],cfg,role='base')


def test_final_selection_is_bound_to_checkpoint_and_summary(tmp_path):
    from maniskill_myws.pld.libero_selection import select_source_validation,require_source_selection
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    p=tmp_path/'evalrun';(p/'eval').mkdir(parents=True)
    alignment=tmp_path/'alignment.json';alignment.write_text('{}')
    checkpoint=tmp_path/'specialist.pt';checkpoint.write_bytes(b'weights')
    (p/'config.json').write_text(json.dumps(dict(cfg,command_options=dict(alignment_manifest=str(alignment)))))
    (p/'metadata.json').write_text(json.dumps({'status':'COMPLETED'}))
    row=dict(source=Protocol(cfg).source,target=Protocol(cfg).source,distance='D0',
        evaluation_scope='source_validation',evaluation_policy='otf',seeds=cfg['validation_env_seeds'],episodes=20,
        SR_residual=.4,residual_successes=8,alignment_sha256=file_sha256(alignment),
        residual_checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint))
    summary=p/'eval/summary.json';summary.write_text(json.dumps([row]))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(select_source_validation([p],cfg,role='residual')))
    require_source_selection(selection,cfg,alignment,checkpoint)
    original=selection.read_text()
    edited=json.loads(original);edited['candidates'][0]['success_rate']=.9
    selection.write_text(json.dumps(edited))
    with pytest.raises(ValueError):require_source_selection(selection,cfg,alignment,checkpoint)
    selection.write_text(original)
    checkpoint.write_bytes(b'replaced')
    with pytest.raises(ValueError):require_source_selection(selection,cfg,alignment,checkpoint)
    checkpoint.write_bytes(b'weights');row['SR_residual']=.5;row['residual_successes']=10
    summary.write_text(json.dumps([row]))
    with pytest.raises(ValueError):require_source_selection(selection,cfg,alignment,checkpoint)


def test_selection_rejects_different_execution_contract(tmp_path):
    from maniskill_myws.pld.libero_selection import select_source_validation
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    p=tmp_path/'run';(p/'eval').mkdir(parents=True)
    manifest=p/'alignment.json';manifest.write_text('{}')
    (p/'metadata.json').write_text(json.dumps(dict(status='COMPLETED')))
    changed=dict(cfg,replan_steps=1,command_options=dict(alignment_manifest=str(manifest)))
    (p/'config.json').write_text(json.dumps(changed))
    row=dict(source=Protocol(cfg).source,target=Protocol(cfg).source,distance='D0',evaluation_scope='source_validation',
        seeds=cfg['validation_env_seeds'],episodes=20,SR_base=.5,base_successes=10,alignment_sha256=file_sha256(manifest))
    (p/'eval/summary.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='execution'):select_source_validation([p],cfg,role='base')


def test_whole_episode_active_checkpoint_can_be_validated():
    from maniskill_myws.pld.libero_protocol import require_specialist_regimen,residual_training_spec
    cfg=json.loads(Path('configs/pld_libero/anchor_bowl_v2.json').read_text())
    meta=dict(training_spec=residual_training_spec(cfg),training_steps=27060,active_steps=5060,sac_config={})
    require_specialist_regimen(meta,cfg,source_validation=True)
    with pytest.raises(ValueError):require_specialist_regimen(dict(meta,training_steps=999999),cfg,source_validation=True)
