"""Publish the final HF model card and executed notebook after the full audit."""
import json
import os
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','1')
from data import ROOT,HF_REPO,TRAIN_SETS,TASKS
from artifacts import verified_final
from report import collect_matrix
from prepare import write_json,sha256


def publish_card():
    import nbformat
    from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationDelete,hf_hub_download
    cells,provenance=collect_matrix()
    notebook_path=ROOT/'notebooks/02_turbovla_multitask_eval.ipynb'
    notebook=nbformat.read(notebook_path,as_version=4)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    if (not code_cells or any(cell.execution_count is None for cell in code_cells)
        or any(output.output_type=='error' for cell in code_cells for output in cell.outputs)):
        raise ValueError('Execute the complete notebook successfully before final HF publication')
    api=HfApi()
    publications={}
    initial_hashes=set()
    for variant in TRAIN_SETS:
        _,_,metadata=verified_final(variant)
        initial_hashes.add(metadata['initial_state_sha256'])
        record=json.loads((ROOT/'docs/turbovla_multitask/publications'/f'{variant}.json').read_text())
        if record['verified_download'] is not True or record['checkpoint_sha256']!=metadata['checkpoint_sha256']:
            raise ValueError(f'Model publication evidence differs: {variant}')
        remote=api.model_info(HF_REPO,revision=record['revision'],files_metadata=True)
        file=next(f for f in remote.siblings if f.rfilename==f'{variant}/final.pth')
        if file.lfs.sha256!=metadata['checkpoint_sha256']:
            raise ValueError(f'Remote checkpoint differs: {variant}')
        publications[variant]=record
    if len(initial_hashes)!=1: raise ValueError('All three original initializations must match')
    lookup={(cell['variant'],cell['task']):cell for cell in cells}
    lines=['---','tags:','- robotics','- vision-language-action','- libero','- turbovla',
        'license: other','license_name: turbovla-and-dinov3',
        f'license_link: https://huggingface.co/{HF_REPO}/blob/main/A/DINOv3_LICENSE.md',
        '---','# TurboVLA: nested-task generalization','',
        'Three independently trained policies, executed in order on one RTX4090:',
        '**A: D0; B: D0+D1; C: D0+D1+D2.** All start from identical original',
        'DINOv3 ViT-B/16 +BERT+GroundingDINO initialization. The released all-LIBERO',
        'policy is not used for initialization. H1/H2 demonstrations and statistics',
        'are excluded from every training run.','',
        'Each checkpoint contains the complete raw and EMA policy state dicts,',
        'including both backbones. Primary evaluation uses EMA. BERT is frozen;',
        'DINOv3 and remaining policy modules are trained for3001 AdamW updates',
        '(batch8,LR5e-5,seed0,EMA0.999).','',
        '## Models','',
        '| Variant | Training tasks | Verified model revision |',
        '|---|---|---|']
    for variant,tasks in TRAIN_SETS.items():
        rev=publications[variant]['revision']
        lines.append(f'| {variant} | {"+".join(tasks)} | [{rev[:12]}](https://huggingface.co/{HF_REPO}/tree/{rev}/{variant}) |')
    lines+=['','Each model folder includes `final.pth`, its own `stats.json`, exact',
        'training configuration/log, source/initialization hashes and licenses.',
        '`evaluation/` includes75 recorded episodes and dual-camera videos.','',
        '| Task | Suite | Instruction |','|---|---|---|']
    for task,spec in TASKS.items():
        lines.append(f"| {task} | {spec['suite']} | {spec['name'].replace('_',' ')} |")
    lines+=[
        '## Results','',
        'Successes out of15 episodes per cell; seeds10000–10014 are shared across',
        'models.10 settling steps,12-action chunks,256px images, reference horizons',
        '220(Spatial) and280(Goal/Object). Success is the LIBERO simulator predicate.','',
        '| Train set | D0 | D1 | D2 | H1 | H2 |','|---|---|---|---|---|---|']
    for variant,tasks in TRAIN_SETS.items():
        values=[f"{lookup[variant,task]['successes']}/15" for task in TASKS]
        lines.append('| '+'+'.join(tasks)+' | '+' | '.join(values)+' |')
    lines+=['','![Generalization matrix](generalization_matrix.png)','',
        '`generalization_matrix.csv` retains seen/unseen labels, exact rates and',
        'Wilson95% confidence intervals. With15 episodes, one success changes the',
        'rate by6.67 percentage points; these results do not establish universal',
        'zero-shot performance.','',
        '## Load and evaluate','',
        'The full executed walkthrough is [02_turbovla_multitask_eval.ipynb](02_turbovla_multitask_eval.ipynb).',
        'Use the experiment branch and its pinned environment:',
        'https://github.com/qrvmil/maniskill_myws/tree/exp/turbovla-multitask-generalization','',
        '```bash','export HF_HOME=/workspace/.hf_home','bash turbovla_multitask/setup.sh',
        '```','',
        'Select A/B/C in the notebook. It downloads model and statistics from the',
        'verified revision, checks SHA256, loads the complete TurboVLAPolicy,',
        'restores both encoders from those full weights, connects LIBERO and',
        'executes15 seeded episodes. This private repository',
        'requires an authenticated HF token.','',
        '## Scope and provenance','',
        'Data: pinned `openvla/modified_libero_rlds`;46/47/48 filtered demonstrations',
        'for D0/D1/D2. Images are already rotated256px no-noop replays; RLDS',
        'observation[t] aligns with action[t]. Statistics are fit to each own corpus.',
        'These data differ from the pi0 comparison’s50 native128px demonstrations.',
        'Our3001-update/batch8 experiment also differs from the paper’s joint',
        '40k-update/batch128 recipe; it does not reproduce the paper’s aggregate SR.','',
        'Original ViT-B/16 is obtained from a pinned public NYU mirror. Full checksum,',
        'strict Meta/Transformers loads and forward parity are recorded in each',
        '`dinov3_conversion.json`. The small deterministic RoPE rounding difference',
        'between Meta and Transformers is documented there. All three runs use',
        'the same converted initialization.','',
        'Upstream TurboVLA: https://github.com/H-EmbodVis/TurboVLA',
        'Paper: https://arxiv.org/abs/2607.27205',
        'Retained DINOv3 license is bundled with each model; consult it and upstream',
        'project licenses for applicable terms.']
    card=('\n'.join(lines)+'\n').encode()
    card_path=ROOT/'docs/turbovla_multitask/HF_MODEL_CARD.md';card_path.write_bytes(card)
    files={'README.md':card_path,'generalization_matrix.png':ROOT/'docs/turbovla_multitask/generalization_matrix.png',
           'generalization_matrix.csv':ROOT/'docs/turbovla_multitask/generalization_matrix.csv',
           '02_turbovla_multitask_eval.ipynb':notebook_path}
    operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in files.items()]
    if 'experiment_status.txt' in api.list_repo_files(HF_REPO):
        operations.append(CommitOperationDelete(path_in_repo='experiment_status.txt'))
    info=api.create_commit(HF_REPO,operations=operations,commit_message='Publish complete verified experiment card, matrix and executed evaluation notebook')
    hashes={name:sha256(path) for name,path in files.items()}
    for name,expected in hashes.items():
        if sha256(hf_hub_download(HF_REPO,name,revision=info.oid))!=expected:
            raise ValueError(f'Final HF artifact differs: {name}')
    write_json(ROOT/'docs/turbovla_multitask/model_card_publication.json',
               dict(repo=HF_REPO,revision=info.oid,hashes=hashes,verified_download=True))
    print('FINAL_HF_ARTIFACTS_VERIFIED',info.oid,flush=True)


if __name__=='__main__':publish_card()
