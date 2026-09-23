"""Execute cached notebook mode with clearly synthetic evidence in a temporary tree."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_cached_notebook_executes_all_cells_without_gpu(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    from jupyter_client.kernelspec import KernelSpecManager
    if 'pi0-multitask' not in KernelSpecManager().find_kernel_specs():
        pytest.skip('Requires the documented pinned notebook kernel')
    from maniskill_myws.pld.multitask_protocol import TASKS, TRAIN_SETS, SEEDS, UPDATES, prompt, action_changes
    from maniskill_myws.pld.multitask_report import export

    repository = Path(__file__).resolve().parents[1]
    # The notebook discovers this temporary repository; source imports are read-only.
    (tmp_path / 'src').symlink_to(repository / 'src', target_is_directory=True)
    output = tmp_path / 'docs/multitask_sft'
    evidence = output / 'evidence'

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')

    for variant in TRAIN_SETS:
        for updates in UPDATES:
            checkpoint = f'/synthetic-smoke-only/{variant}/{updates}'
            saved = dict(optimizer_updates=updates, params_files={'fixture': 'synthetic'},
                         normalization_sha256='synthetic-normalizer')
            write(evidence / 'training' / variant / f'update_{updates}.json', saved)
            for task in TASKS if updates == 3001 else ('D0', 'D1', 'D2'):
                folder = evidence / 'eval' / variant / str(updates) / task
                rows = [dict(variant=variant, task=task, checkpoint=checkpoint,
                             seen=task in TRAIN_SETS[variant], seed=seed,
                             prompt=prompt(task), reset_hash=f'synthetic-{task}-{seed}',
                             success=seed % 2 == 0, length=10, ever_reached_10cm=True,
                             task_progress=dict(ever_grasped=False, ever_lifted_3cm=False))
                        for seed in SEEDS]
                binding = dict(variant=variant, task=task, checkpoint=checkpoint, seeds=list(SEEDS),
                               checkpoint_params=saved['params_files'],
                               normalization_sha256=saved['normalization_sha256'],
                               normalization_path='/synthetic-smoke-only/norm_stats.json',
                               action_contract='SYNTHETIC FIXTURE', action_horizon=50, replan_steps=5)
                write(folder / 'binding.json', binding)
                write(folder / 'episodes.json', rows)
                write(folder / 'complete.json', dict(synthetic_fixture=True))
        correct = np.zeros((20, 50, 7), np.float32)
        shuffled = np.full_like(correct, .1)
        sensitivity = [dict(variant=variant, task=task, seed=seed,
                            donor_seed=SEEDS[(i + 1) % 10], bank_sha256=f'synthetic-{task}',
                            noise_sha256=f'synthetic-{seed}',
                            **action_changes(correct[0], shuffled[0]))
                       for task in ('D0', 'H1') for i, seed in enumerate(SEEDS[:10])]
        folder = evidence / 'eval' / variant / '3001'
        write(folder / 'image_sensitivity.json', sensitivity)
        np.savez_compressed(folder / 'image_sensitivity_actions.npz', correct=correct,
                            shuffled=shuffled, tasks=[r['task'] for r in sensitivity],
                            seeds=[r['seed'] for r in sensitivity])

    from maniskill_myws.pld.multitask_report import read_results
    interrupted = evidence / 'eval/B/500/D0/complete.json'
    interrupted.unlink()  # Partial rows exist but must not enter the N50 analysis.
    with pytest.raises(ValueError, match='Incomplete evaluation'):
        read_results(evidence / 'eval')
    episodes, summaries, _ = read_results(evidence / 'eval', allow_incomplete_trajectory=True)
    assert 'D0' not in episodes['B'][500]
    assert len(summaries) == 50
    final_marker = evidence / 'eval/C/3001/H1/complete.json'
    final_marker.unlink()
    with pytest.raises(ValueError, match='Incomplete evaluation'):
        read_results(evidence / 'eval', allow_incomplete_trajectory=True)
    write(final_marker, dict(synthetic_fixture=True))
    export(evidence / 'eval', output, allow_incomplete_trajectory=True)
    notebook = nbformat.read(repository / 'notebooks/01_multitask_sft_eval.ipynb', as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
        '# SYNTHETIC EXECUTION TEST — NOT EXPERIMENT RESULTS\nAll observations are artificial fixtures.'))
    parameters = next(c for c in notebook.cells if 'parameters' in c.metadata.get('tags', []))
    parameters.source += f'\nEXPERIMENT_ROOT = Path({str(tmp_path / "runtime")!r})\n'
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    client = nbclient.NotebookClient(notebook, timeout=120, kernel_name='pi0-multitask',
                                    resources={'metadata': {'path': str(tmp_path)}})
    client.execute()
    nbformat.write(notebook, tmp_path / 'SYNTHETIC_ONLY_executed.ipynb')
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
    assert any('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
