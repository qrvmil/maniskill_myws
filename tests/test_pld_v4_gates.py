"""Registered adaptation staging must not accept cherry-picked candidates."""
import pytest
from test_pld_v4 import selection_fixture
from maniskill_myws.pld.libero_adaptation_selection import select_d1_validation

def test_selection_requires_complete_registered_staging(tmp_path):
    cfg,run,_,_=selection_fixture(tmp_path)
    with pytest.raises(ValueError,match='stage|milestone'):
        select_d1_validation([run],cfg,role='residual',policy='deterministic_actor')


def test_selection_rejects_missing_duplicate_or_off_schedule_candidates(tmp_path):
    import json
    from test_pld_v4 import complete_selection_fixture
    cfg,runs,_,_,stage=complete_selection_fixture(tmp_path)
    for candidates in (runs[:-1],runs+[runs[0]]):
        with pytest.raises(ValueError):
            select_d1_validation(candidates,cfg,role='residual',policy='deterministic_actor',stage_decision=stage)
    checkpoint=tmp_path/'residual.pt'
    provenance=checkpoint.with_suffix('.json');data=json.loads(provenance.read_text())
    data['active_steps']=0;data['training_steps']=22000;provenance.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='milestone'):
        select_d1_validation(runs,cfg,role='residual',policy='deterministic_actor',stage_decision=stage)


def test_stage_rejects_unverified_continuation_and_requires_100k_when_learning(tmp_path):
    import json
    from test_pld_v4 import complete_selection_fixture
    from maniskill_myws.pld.libero_adaptation_selection import require_d1_stage,d1_stage_record
    from maniskill_myws.pld.libero_protocol import paired_summary
    cfg,runs,_,_,stage=complete_selection_fixture(tmp_path)
    original=json.loads(stage.read_text());tampered=dict(original,decision='continue')
    stage.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='continuation'):require_d1_stage(stage,cfg)
    pairpath=runs[-1]/'eval'/f"{cfg['residual_training_task']['name']}_episodes.json"
    pairs=json.loads(pairpath.read_text());pairs['residual'][10]['success']=True
    pairpath.write_text(json.dumps(pairs))
    path=runs[-1]/'eval/summary.json';rows=json.loads(path.read_text())
    rows[0].update(paired_summary(pairs['base'],pairs['residual']));path.write_text(json.dumps(rows))
    record=d1_stage_record(runs,cfg);assert record['decision']=='continue'
    stage.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='complete'):
        select_d1_validation(runs,cfg,role='residual',policy='deterministic_actor',stage_decision=stage)


def test_base_replay_cannot_weaken_success_count_or_change_collection(tmp_path):
    import json
    from test_pld_v4 import config,alignment,base_replay_fixture
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    from maniskill_myws.pld.libero_experiment import _load_offline
    cfg=config();cfg['rl_image_size']=4;protocol=Protocol(cfg);manifest,_=alignment(tmp_path,cfg)
    path,replay,metadata=base_replay_fixture(tmp_path,cfg,protocol,manifest)
    replay.save(path,**dict(metadata,successes=49))
    with pytest.raises(ValueError,match='50 distinct'):_load_offline(path,cfg,protocol,manifest)
    from pathlib import Path
    collection=Path(metadata['collection_path']);data=json.loads(collection.read_text());data['episodes'][0]['seed']=8000
    collection.write_text(json.dumps(data));metadata['collection_sha256']=file_sha256(collection)
    replay.save(path,**metadata)
    with pytest.raises(ValueError,match='50 distinct'):_load_offline(path,cfg,protocol,manifest)


def test_final_short_run_rejected_with_frozen_selection_disclosure(tmp_path):
    import json
    from types import SimpleNamespace
    from test_pld_v4 import complete_selection_fixture
    from maniskill_myws.pld.libero_protocol import Protocol,file_sha256
    from maniskill_myws.pld.libero_experiment import evaluate
    cfg,runs,manifest,checkpoint,stage=complete_selection_fixture(tmp_path)
    record=select_d1_validation(runs,cfg,role='residual',policy='deterministic_actor',stage_decision=stage)
    selected=tmp_path/'selected.json';selected.write_text(json.dumps(record))
    args=SimpleNamespace(mode='eval',validation=False,episodes=2,selection_manifest=selected,
        alignment_manifest=manifest,checkpoint=checkpoint,eval_policy='deterministic_actor')
    run=SimpleNamespace(path=tmp_path,meta={})
    with pytest.raises(ValueError,match='all 50'):
        evaluate(cfg,args,run,None,None,Protocol(cfg),None)
    assert json.loads((tmp_path/'frozen_selection.json').read_text())==record
    assert run.meta['selection_manifest_sha256']==file_sha256(selected)
    assert run.meta['d1_improvement_gate_passed'] is True
