"""Source-only checkpoint selection and immutable manifests for SFT checkpoints."""
import json
from pathlib import Path
from .libero_protocol import Protocol,file_sha256,directory_manifest,residual_training_spec


def select_source_validation(evaluations,cfg,*,role,policy='otf'):
    if role not in ('base','residual'):raise ValueError('Unknown selection role')
    protocol=Protocol(cfg);candidates=[]
    for path in evaluations:
        path=Path(path).resolve()
        if json.loads((path/'metadata.json').read_text()).get('status')!='COMPLETED':
            raise ValueError('Selection requires completed evaluation')
        summary=path/'eval/summary.json';rows=json.loads(summary.read_text())
        if len(rows)!=1:raise ValueError('Selection must contain source only')
        row=rows[0]
        if (row.get('source')!=protocol.source or row.get('target')!=protocol.source or row.get('distance')!='D0'
            or row.get('evaluation_scope')!='source_validation' or row.get('seeds')!=cfg['validation_env_seeds']
            or row.get('episodes')!=len(cfg['validation_env_seeds'])):
            raise ValueError('Selection requires full registered D0 validation, never final/unseen metrics')
        evaluated_config=json.loads((path/'config.json').read_text())
        evaluated_protocol=Protocol(evaluated_config)
        if evaluated_protocol.split_hash!=protocol.split_hash or evaluated_protocol.execution_hash!=protocol.execution_hash:
            raise ValueError('Evaluation split/execution contract differs from selection')
        if role=='residual' and residual_training_spec(evaluated_config)!=residual_training_spec(cfg):
            raise ValueError('Evaluation training regimen differs from selection')
        command=evaluated_config['command_options']
        alignment=Path(command['alignment_manifest']).resolve()
        if file_sha256(alignment)!=row['alignment_sha256']:raise ValueError('Alignment changed after validation')
        checkpoint=row.get('residual_checkpoint') if role=='residual' else None
        if role=='residual':
            if policy=='auto':
                if row.get('evaluation_policy') not in ('deterministic_actor','otf'):
                    raise ValueError('Source deployment selection permits only deterministic or exact OTF')
            elif row.get('evaluation_policy')!=policy:raise ValueError('Cannot mix policy modes for selection')
            if not checkpoint or file_sha256(checkpoint)!=row['checkpoint_sha256']:raise ValueError('Residual checkpoint changed')
        rate=row['SR_base' if role=='base' else 'SR_residual']
        successes=row['base_successes' if role=='base' else 'residual_successes']
        if rate!=successes/len(cfg['validation_env_seeds']):raise ValueError('Inconsistent success count')
        candidates.append(dict(selected_evaluation=str(path),summary_sha256=file_sha256(summary),
            alignment_manifest=str(alignment),alignment_sha256=row['alignment_sha256'],checkpoint=checkpoint,
            checkpoint_sha256=row.get('checkpoint_sha256') if checkpoint else None,success_rate=rate))
        if role=='residual' and policy=='auto':
            magnitude=row['executed_residual_mean_abs']
            if not __import__('math').isfinite(magnitude) or magnitude<0:
                raise ValueError('Invalid selection correction magnitude')
            provenance_path=Path(checkpoint).with_suffix('.json')
            provenance=json.loads(provenance_path.read_text())
            steps=provenance['training_steps']
            if type(steps) is not int or steps<0 or provenance['checkpoint_sha256']!=row['checkpoint_sha256']:
                raise ValueError('Invalid source checkpoint age/provenance')
            candidates[-1].update(deployment_policy=row['evaluation_policy'],executed_residual_mean_abs=magnitude,
                training_steps=steps,checkpoint_provenance_sha256=file_sha256(provenance_path))
    if not candidates:raise ValueError('No completed source candidates')
    if role=='residual' and len({x['alignment_sha256'] for x in candidates})!=1:
        raise ValueError('Residual selection requires one frozen base')
    # Ties preserve caller's predeclared checkpoint order (typically earliest first).
    if role=='residual' and policy=='auto':
        chosen=max(candidates,key=lambda x:(x['success_rate'],-x['executed_residual_mean_abs'],-x['training_steps']))
        return dict(role=role,policy=chosen['deployment_policy'],selection_policy='auto',source=protocol.source,
            split_hash=protocol.split_hash,validation_seeds=cfg['validation_env_seeds'],candidates=candidates,
            tie_rule='smaller mean executed correction, then earliest recorded training step',**chosen)
    chosen=max(candidates,key=lambda x:x['success_rate'])
    return dict(role=role,policy=policy if role=='residual' else 'base',source=protocol.source,
        split_hash=protocol.split_hash,validation_seeds=cfg['validation_env_seeds'],
        candidates=candidates,tie_rule='first in supplied chronological checkpoint order',**chosen)


def require_source_selection(path,cfg,alignment_manifest,checkpoint):
    record=json.loads(Path(path).read_text())
    if record.get('role')!='residual':raise ValueError('Final residual evaluation requires residual selection')
    paths=[c['selected_evaluation'] for c in record['candidates']]
    verified=select_source_validation(paths,cfg,role='residual',policy=record.get('selection_policy',record['policy']))
    if record!=verified:raise ValueError('Source selection candidates or chosen result changed')
    for key in ('source','split_hash','validation_seeds','summary_sha256','checkpoint_sha256','alignment_sha256'):
        if record[key]!=verified[key]:raise ValueError('Source selection evidence changed')
    if record['alignment_sha256']!=file_sha256(alignment_manifest) or record['checkpoint_sha256']!=file_sha256(checkpoint):
        raise ValueError('Final evaluation checkpoint differs from source selection')
    return record


def intermediate_alignment_manifests(final_manifest,output):
    manifest=json.loads(Path(final_manifest).read_text())
    root=Path(manifest['aligned_checkpoint']).parent
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=[]
    for checkpoint in sorted(root.iterdir(),key=lambda p:int(p.name) if p.name.isdecimal() else -1):
        if not checkpoint.is_dir() or not checkpoint.name.isdecimal():continue
        normalizer=checkpoint/'assets'/manifest['repo_id']/'norm_stats.json'
        if not normalizer.exists():continue
        updates=int(checkpoint.name)+(0 if manifest['method'] in ('full_torch','full_cpu') else 1)
        if updates>manifest['alignment_steps']:raise ValueError('Checkpoint exceeds recorded SFT budget')
        item=dict(manifest,alignment_steps=updates,aligned_checkpoint=str(checkpoint.resolve()),
                  checkpoint_files=directory_manifest(checkpoint),
                  normalization=dict(path=str(normalizer.resolve()),sha256=file_sha256(normalizer)))
        path=output/f'alignment_{updates}.json';path.write_text(json.dumps(item,indent=2)+'\n');result.append(str(path))
    return result


def rebind_alignment_split(path, old_config, new_config):
    """Reuse verified source-only weights after preregistering new held-out seeds.

    This creates lineage, not another SFT run. Training data/tasks and action
    preprocessing cannot change through this operation.
    """
    old,new=Protocol(old_config),Protocol(new_config)
    manifest=old.require_alignment(path)
    for key in ('source','tasks','train_env_seeds'):
        if old_config[key]!=new_config[key]:
            raise ValueError('Cannot rebind training tasks/seeds')
    if old.execution_hash!=new.execution_hash:
        raise ValueError('Cannot rebind action/observation execution contract')
    return dict(manifest,split_hash=new.split_hash,parent_alignment=dict(
        path=str(Path(path).resolve()),sha256=file_sha256(path),split_hash=old.split_hash),
        split_rebinding_reason='V3 preregistered held-out seeds; source-only weights and training split unchanged')
