"""Execute the complete tutorial in a fresh kernel and save notebook/HTML evidence."""
import os
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from data import ROOT
from report import collect_matrix
from prepare import write_json,sha256


def execute():
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    cells,_=collect_matrix()
    if len(cells)!=15 or sum(c['episodes'] for c in cells)!=225:
        raise ValueError('Require the completed full experiment before final notebook execution')
    path=ROOT/'notebooks/02_turbovla_multitask_eval.ipynb'
    notebook=nbformat.read(path,as_version=4)
    if 'Черновик' in notebook.cells[0].source:
        raise ValueError('Regenerate the final notebook after the verified matrix exists')
    client=NotebookClient(notebook,timeout=600,kernel_name='turbovla-multitask',
        resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
    errors=[output for cell in code_cells for output in cell.outputs if output.output_type=='error']
    if errors or any(cell.execution_count is None for cell in code_cells):
        raise ValueError('Notebook did not execute every code cell successfully')
    temporary=path.with_suffix('.ipynb.tmp')
    nbformat.write(notebook,temporary);temporary.replace(path)
    html,_=HTMLExporter(template_name='lab').from_notebook_node(notebook)
    preview=ROOT/'docs/turbovla_multitask/preview/02_turbovla_multitask_eval.html'
    preview.parent.mkdir(parents=True,exist_ok=True);preview.write_text(html)
    status=dict(code_cells=len(code_cells),executed_cells=sum(c.execution_count is not None for c in code_cells),
        errors=len(errors),full_matrix_episodes=225,new_notebook_rollouts=15,
        fresh_kernel=True,notebook_sha256=sha256(path),html_sha256=sha256(preview),
        visual_inspection_pending=True)
    write_json(ROOT/'docs/turbovla_multitask/notebook_execution.json',status)
    print('NOTEBOOK_EXECUTED',status,flush=True)


if __name__=='__main__':execute()
