"""Recompute transfer metrics from completed paired episode artifacts."""
import json
from pathlib import Path
import numpy as np
from .libero_protocol import Protocol,file_sha256,paired_summary,task_key,require_specialist_regimen


def gather_transfer_results(paths):
    rows=[];seen=set();specialists={};missing={}
    for path in map(Path,paths):
        meta=json.loads((path/'metadata.json').read_text())
        cfg=json.loads((path/'config.json').read_text())
        args=cfg['command_options'];protocol=Protocol(cfg)
        if meta['status']!='COMPLETED' or args['mode']!='eval':
            raise ValueError('Only completed learned-residual evaluation runs can be summarized')
        checkpoint=Path(args['checkpoint'])
        provenance=json.loads(checkpoint.with_suffix('.json').read_text())
        require_specialist_regimen(provenance,cfg)
        if (provenance['checkpoint_sha256']!=file_sha256(checkpoint)
            or provenance['alignment_sha256']!=file_sha256(args['alignment_manifest'])
            or provenance['source']!=protocol.source
            or provenance['split_hash']!=protocol.split_hash
            or provenance['execution_hash']!=protocol.execution_hash
            or provenance['training_seed']!=cfg['training_seed']):
            raise ValueError('Evaluation specialist provenance mismatch')
        key=(protocol.source,cfg['training_seed'])
        digest=provenance['checkpoint_sha256']
        if key in specialists and specialists[key]!=digest:
            raise ValueError('Cannot mix different specialist checkpoints across a distance ladder')
        specialists[key]=digest
        missing.setdefault(key,{task_key(t) for t in cfg['tasks']})
        task_map={task_key(t):t for t in cfg['tasks']}
        for recorded in json.loads((path/'eval/summary.json').read_text()):
            for field in ('training_steps','training_spec','sac_config'):
                if recorded.get('residual_'+field)!=provenance[field]:
                    raise ValueError('Evaluation recorded a different residual training regimen')
            if recorded.get('checkpoint_sha256')!=digest or recorded.get('alignment_sha256')!=provenance['alignment_sha256']:
                raise ValueError('Evaluation recorded a different checkpoint/base than the current files')
            task=task_map[recorded['target']]
            pair=json.loads((path/'eval'/f"{task['name']}_episodes.json").read_text())
            values=paired_summary(pair['base'],pair['residual'])
            if not set(values['seeds']).issubset(cfg['eval_seeds']):
                raise ValueError('Final transfer summary contains non-evaluation seeds')
            if any(not np.isclose(recorded[k],values[k]) for k in ['SR_base','SR_residual','delta_SR']):
                raise ValueError('Recorded metrics do not match raw paired episodes')
            identity=(*key,recorded['target'])
            if identity in seen:raise ValueError('Duplicate source/seed/target evaluation')
            seen.add(identity);missing[key].discard(recorded['target'])
            rows.append(dict(source=protocol.source,target=task_key(task),distance=task['distance'],
                training_seed=cfg['training_seed'],residual_training_steps=provenance['training_steps'],
                residual_training_spec=provenance['training_spec'],residual_sac_config=provenance['sac_config'],
                checkpoint=str(checkpoint),run=str(path),**values))
    if not rows:raise ValueError('No completed paired results')
    buckets=[]
    for source,seed,distance in sorted({(r['source'],r['training_seed'],r['distance']) for r in rows}):
        subset=[r for r in rows if (r['source'],r['training_seed'],r['distance'])==(source,seed,distance)]
        buckets.append(dict(source=source,training_seed=seed,distance=distance,tasks=len(subset),
            mean_gain=float(np.mean([r['delta_SR'] for r in subset]))))
    return dict(tasks=rows,buckets=buckets,
        missing_tasks=[dict(source=k[0],training_seed=k[1],targets=sorted(v)) for k,v in missing.items()],
        interpretation='Exploratory per-training-seed results; episode bootstrap does not measure training-seed variance.')
