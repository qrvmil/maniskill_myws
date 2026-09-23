"""Frozen D1-only residual selection; D0 retention never enters ranking."""
import hashlib
import json
import math
from pathlib import Path

from .libero_protocol import (Protocol, file_sha256, paired_summary,
                              residual_training_spec, require_specialist_regimen)


def config_hash(cfg):
    clean={k:v for k,v in cfg.items() if k!='command_options'}
    return hashlib.sha256(json.dumps(clean,sort_keys=True).encode()).hexdigest()


def _validated_candidates(evaluations,cfg,*,role,policy):
    protocol=Protocol(cfg)
    if not protocol.is_adaptation or role!='residual' or policy!='deterministic_actor':
        raise ValueError('V4 selects only the preregistered deterministic D1 residual')
    candidates=[]
    for entry in evaluations:
        path=Path(entry).resolve()
        if json.loads((path/'metadata.json').read_text()).get('status')!='COMPLETED':
            raise ValueError('Selection requires completed D1 validation')
        evaluated=json.loads((path/'config.json').read_text())
        if config_hash(evaluated)!=config_hash(cfg):
            raise ValueError('Validation config differs from frozen experiment')
        command=evaluated['command_options']
        if (command.get('mode')!='eval' or command.get('validation') is not True
                or command.get('distance')!='D1'
                or (command.get('eval_policy') or evaluated['eval_residual'])!=policy):
            raise ValueError('Selection requires D1 validation commands only')
        rows=json.loads((path/'eval/summary.json').read_text())
        if len(rows)!=1:
            raise ValueError('Selection requires exactly one D1 task')
        row=rows[0]
        if (row.get('source')!=protocol.residual_training_key
                or row.get('target')!=protocol.residual_training_key
                or row.get('distance')!='D1' or row.get('evaluation_scope')!='residual_validation'
                or row.get('evaluation_policy')!=policy
                or row.get('seeds')!=cfg['validation_env_seeds']):
            raise ValueError('Selection requires registered D1 validation, never D0/final metrics')
        pairs_path=path/'eval'/f"{protocol.residual_training_task['name']}_episodes.json"
        pairs=json.loads(pairs_path.read_text())
        summary=paired_summary(pairs['base'],pairs['residual'])
        if any(row.get(k)!=v for k,v in summary.items()):
            raise ValueError('Validation summary differs from paired episode evidence')
        manifest=Path(command['alignment_manifest']).resolve()
        protocol.require_alignment(manifest)
        checkpoint=Path(command['checkpoint']).resolve()
        if (str(checkpoint)!=row.get('residual_checkpoint')
                or file_sha256(checkpoint)!=row.get('checkpoint_sha256')
                or file_sha256(manifest)!=row.get('alignment_sha256')):
            raise ValueError('Validation checkpoint or alignment changed')
        provenance_path=checkpoint.with_suffix('.json')
        provenance=json.loads(provenance_path.read_text())
        require_specialist_regimen(provenance,cfg,source_validation=True)
        expected=dict(source=protocol.residual_training_key,
            base_alignment_task=protocol.base_alignment_key,residual_training_task=protocol.residual_training_key,
            split_hash=protocol.split_hash,execution_hash=protocol.execution_hash,
            training_seed=cfg['training_seed'],checkpoint_sha256=row['checkpoint_sha256'],
            alignment_sha256=row['alignment_sha256'])
        if any(provenance.get(k)!=v for k,v in expected.items()):
            raise ValueError('D1 checkpoint role/provenance mismatch')
        magnitude=row['executed_residual_mean_abs']
        if not math.isfinite(magnitude) or magnitude<0:
            raise ValueError('Invalid correction magnitude')
        candidates.append(dict(selected_evaluation=str(path),
            summary_sha256=file_sha256(path/'eval/summary.json'),
            pairs_sha256=file_sha256(pairs_path),
            evaluation_config_sha256=file_sha256(path/'config.json'),
            checkpoint=str(checkpoint),checkpoint_sha256=row['checkpoint_sha256'],
            checkpoint_provenance_sha256=file_sha256(provenance_path),
            alignment_manifest=str(manifest),alignment_sha256=row['alignment_sha256'],
            success_rate=row['SR_residual'],base_success_rate=row['SR_base'],
            validation_gain=row['delta_SR'],executed_residual_mean_abs=magnitude,
            training_steps=provenance['training_steps'],active_steps=provenance['active_steps']))
    if not candidates or len({c['alignment_sha256'] for c in candidates})!=1:
        raise ValueError('Selection needs D1 candidates sharing one frozen D0 base')
    if len({c['selected_evaluation'] for c in candidates})!=len(candidates):
        raise ValueError('Duplicate selection evidence')
    milestones=cfg['checkpoint_active_steps']
    horizon=protocol.residual_training_task['horizon']
    for candidate in candidates:
        matches=[m for m in milestones if m <= candidate['active_steps'] < m+horizon]
        if len(matches)!=1:
            raise ValueError('Checkpoint is not a registered active milestone')
        candidate['active_milestone']=matches[0]
    for key in ('checkpoint_sha256','active_steps','active_milestone'):
        if len({c[key] for c in candidates})!=len(candidates):
            raise ValueError('Duplicate milestone/checkpoint evidence')
    return sorted(candidates,key=lambda c:c['active_milestone'])


CONTINUATION_RULE='D1 deterministic validation SR at 50k strictly exceeds 25k'


def d1_stage_record(evaluations,cfg):
    candidates=_validated_candidates(evaluations,cfg,role='residual',policy='deterministic_actor')
    if [c['active_milestone'] for c in candidates]!=[5000,10000,25000,50000]:
        raise ValueError('D1 stage requires all 5k/10k/25k/50k milestones')
    learning=candidates[-1]['success_rate']>candidates[-2]['success_rate']
    last=candidates[-1]
    return dict(evidence=dict(checkpoint=last['checkpoint'],checkpoint_sha256=last['checkpoint_sha256'],active_milestone=50000),
        decision='continue' if learning else 'finish',reason=CONTINUATION_RULE,
        criterion=CONTINUATION_RULE,criterion_passed=learning,config_sha256=config_hash(cfg),
        selection_task=Protocol(cfg).residual_training_key,candidates=candidates)


def require_d1_stage(path,cfg):
    record=json.loads(Path(path).read_text())
    verified=d1_stage_record([c['selected_evaluation'] for c in record['candidates']],cfg)
    if record!=verified:
        raise ValueError('D1 stage evidence or preregistered continuation decision changed')
    return verified


def select_d1_validation(evaluations,cfg,*,role,policy,stage_decision=None):
    candidates=_validated_candidates(evaluations,cfg,role=role,policy=policy)
    if stage_decision is None:
        raise ValueError('Selection requires frozen D1 stage decision')
    stage=require_d1_stage(stage_decision,cfg)
    required=[5000,10000,25000,50000]+([100000] if stage['decision']=='continue' else [])
    if [c['active_milestone'] for c in candidates]!=required or candidates[:4]!=stage['candidates']:
        raise ValueError('Selection requires complete stage-bound milestone evidence')
    protocol=Protocol(cfg)
    chosen=max(candidates,key=lambda c:(c['success_rate'],-c['executed_residual_mean_abs'],-c['training_steps']))
    return dict(role='residual',policy=policy,frozen=True,
        selection_task=protocol.residual_training_key,base_alignment_task=protocol.base_alignment_key,
        source=protocol.residual_training_key,split_hash=protocol.split_hash,
        execution_hash=protocol.execution_hash,config_sha256=config_hash(cfg),
        validation_seeds=cfg['validation_env_seeds'],candidates=candidates,
        stage_decision=str(Path(stage_decision).resolve()),stage_decision_sha256=file_sha256(stage_decision),
        tie_rule='smaller mean executed correction, then earliest recorded training step',
        d1_improvement_gate_passed=chosen['validation_gain']>0,**chosen)


def require_d1_selection(path,cfg,alignment_manifest,checkpoint):
    record=json.loads(Path(path).read_text())
    if record.get('frozen') is not True or record.get('role')!='residual':
        raise ValueError('Final evaluation requires frozen D1 selection')
    verified=select_d1_validation([c['selected_evaluation'] for c in record['candidates']],
                                  cfg,role='residual',policy=record['policy'],stage_decision=record.get('stage_decision'))
    if record!=verified:
        raise ValueError('Frozen D1 selection evidence changed')
    if (file_sha256(alignment_manifest)!=record['alignment_sha256']
            or file_sha256(checkpoint)!=record['checkpoint_sha256']):
        raise ValueError('Final checkpoint differs from frozen D1 selection')
    return record
