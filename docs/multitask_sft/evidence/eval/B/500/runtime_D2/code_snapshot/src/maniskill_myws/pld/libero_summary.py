"""Recompute transfer metrics from completed paired episode artifacts."""
import json
from pathlib import Path
import numpy as np
from .libero_protocol import Protocol,file_sha256,paired_summary,task_key,require_specialist_regimen
from .libero_selection import require_source_selection


def gather_transfer_results(paths):
    rows=[];seen=set();specialists={};missing={};deployments={};seed_blocks={}
    for path in map(Path,paths):
        meta=json.loads((path/'metadata.json').read_text())
        cfg=json.loads((path/'config.json').read_text())
        args=cfg['command_options'];protocol=Protocol(cfg)
        if meta['status']!='COMPLETED' or args['mode']!='eval':
            raise ValueError('Only completed learned-residual evaluation runs can be summarized')
        checkpoint=Path(args['checkpoint'])
        provenance=json.loads(checkpoint.with_suffix('.json').read_text())
        selection=None
        if cfg.get('training_scope'):
            if not args.get('selection_manifest'):
                raise ValueError('Final transfer summary requires verified source selection')
            selection=require_source_selection(args['selection_manifest'],cfg,args['alignment_manifest'],checkpoint)
        disclosure={}
        if protocol.is_adaptation:
            snapshot=path/'frozen_selection.json'
            if selection is None or not snapshot.is_file() or json.loads(snapshot.read_text())!=selection:
                raise ValueError('Final adaptation report requires unchanged frozen selection snapshot')
            disclosure=dict(selection_manifest_sha256=file_sha256(args['selection_manifest']),
                frozen_selection_sha256=file_sha256(snapshot),
                d1_improvement_gate_passed=selection['d1_improvement_gate_passed'],
                selected_d1_validation_gain=selection['validation_gain'],
                final_result_status='improvement_gate_passed' if selection['d1_improvement_gate_passed'] else 'negative_result_gate_failed')
            if any(meta.get(k)!=v for k,v in disclosure.items()):
                raise ValueError('Final metadata does not preserve frozen D1 improvement gate')
        require_specialist_regimen(provenance,cfg,source_validation=selection is not None)
        if (provenance['checkpoint_sha256']!=file_sha256(checkpoint)
            or provenance['alignment_sha256']!=file_sha256(args['alignment_manifest'])
            or provenance['source']!=protocol.residual_training_key
            or provenance['split_hash']!=protocol.split_hash
            or provenance['execution_hash']!=protocol.execution_hash
            or provenance['training_seed']!=cfg['training_seed']):
            raise ValueError('Evaluation specialist provenance mismatch')
        if protocol.is_adaptation and (provenance.get('base_alignment_task')!=protocol.base_alignment_key
                or provenance.get('residual_training_task')!=protocol.residual_training_key):
            raise ValueError('Evaluation task-role provenance mismatch')
        key=(protocol.residual_training_key,cfg['training_seed'])
        digest=provenance['checkpoint_sha256']
        if key in specialists and specialists[key]!=digest:
            raise ValueError('Cannot mix different specialist checkpoints across a distance ladder')
        specialists[key]=digest
        missing.setdefault(key,{task_key(t) for t in protocol.evaluation_tasks})
        task_map={task_key(t):t for t in protocol.evaluation_tasks}
        for recorded in json.loads((path/'eval/summary.json').read_text()):
            if any(recorded.get(k)!=v for k,v in disclosure.items()):
                raise ValueError('Final task summary does not preserve frozen D1 improvement gate')
            for field in ('training_steps','training_spec','sac_config'):
                if recorded.get('residual_'+field)!=provenance[field]:
                    raise ValueError('Evaluation recorded a different residual training regimen')
            if recorded.get('checkpoint_sha256')!=digest or recorded.get('alignment_sha256')!=provenance['alignment_sha256']:
                raise ValueError('Evaluation recorded a different checkpoint/base than the current files')
            policy=recorded.get('evaluation_policy',args.get('eval_policy') or cfg.get('eval_residual','deterministic_actor'))
            if policy not in ('deterministic_actor','otf'):
                raise ValueError('Unsupported deployment policy in transfer summary')
            if selection is not None and policy!=selection['policy']:
                raise ValueError('Summary deployment differs from frozen source selection')
            if key in deployments and deployments[key]!=policy:
                raise ValueError('Cannot mix deployment policies across a distance ladder')
            deployments[key]=policy
            if recorded['target'] not in task_map:
                raise ValueError('Unregistered evaluation task')
            task=task_map[recorded['target']]
            pair=json.loads((path/'eval'/f"{task['name']}_episodes.json").read_text())
            values=paired_summary(pair['base'],pair['residual'])
            if protocol.is_adaptation and values['seeds']!=cfg['eval_seeds']:
                raise ValueError('V4 final results require the complete preregistered seed block')
            if not set(values['seeds']).issubset(cfg['eval_seeds']):
                raise ValueError('Final transfer summary contains non-evaluation seeds')
            if any(not np.isclose(recorded[k],values[k]) for k in ['SR_base','SR_residual','delta_SR']):
                raise ValueError('Recorded metrics do not match raw paired episodes')
            seed_block=tuple(values['seeds'])
            if key in seed_blocks and seed_blocks[key]!=seed_block:
                raise ValueError('Cannot mix paired seed blocks across a distance ladder')
            seed_blocks[key]=seed_block
            identity=(*key,recorded['target'])
            if identity in seen:raise ValueError('Duplicate source/seed/target evaluation')
            seen.add(identity);missing[key].discard(recorded['target'])
            rows.append(dict(source=protocol.residual_training_key,target=task_key(task),distance=task['distance'],
                training_seed=cfg['training_seed'],residual_training_steps=provenance['training_steps'],
                residual_training_spec=provenance['training_spec'],residual_sac_config=provenance['sac_config'],
                checkpoint=str(checkpoint),deployment_policy=policy,run=str(path),**values))
            if protocol.is_adaptation:
                rows[-1].update(base_alignment_task=protocol.base_alignment_key,
                    residual_training_task=protocol.residual_training_key,**disclosure,**gain_interval(values))
    if not rows:raise ValueError('No completed paired results')
    buckets=[]
    for source,seed,distance in sorted({(r['source'],r['training_seed'],r['distance']) for r in rows}):
        subset=[r for r in rows if (r['source'],r['training_seed'],r['distance'])==(source,seed,distance)]
        buckets.append(dict(source=source,training_seed=seed,distance=distance,tasks=len(subset),
            mean_gain=float(np.mean([r['delta_SR'] for r in subset]))))
    return dict(tasks=rows,buckets=buckets,
        missing_tasks=[dict(source=k[0],training_seed=k[1],targets=sorted(v)) for k,v in missing.items()],
        interpretation='Exploratory per-training-seed results; episode bootstrap does not measure training-seed variance.')


def gain_interval(summary):
    """Bootstrap paired gain, with a valid bound when empirical discordance is zero."""
    if summary['residual_only_successes']+summary['base_only_successes']==0:
        bound=1-.05**(1/summary['episodes'])
        return dict(gain_95ci=[-bound,bound],
            gain_interval_method='exact 95% upper bound on zero-observed discordance probability')
    return dict(gain_95ci=summary['paired_bootstrap_95ci'],
        gain_interval_method='paired episode percentile bootstrap, 10000 resamples, seed0')
