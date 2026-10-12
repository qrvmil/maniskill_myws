"""Execute both seminar notebooks in fresh, sequential kernels and save outputs."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from eval_utils import ROOT, write_json

if __name__=='__main__':
    status={}
    for name in ['01_turbovla_step_by_step.ipynb','02_results_and_vram.ipynb']:
        path=ROOT/name
        nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=600,kernel_name='turbovla-libero',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        nbformat.validate(nb)
        errors=[output for cell in nb.cells if cell.cell_type=='code' for output in cell.get('outputs',[]) if output.output_type=='error']
        assert not errors
        nbformat.write(nb,path)
        html,_=HTMLExporter().from_notebook_node(nb)
        preview=ROOT/'preview'/path.with_suffix('.html').name
        preview.parent.mkdir(exist_ok=True)
        preview.write_text(html)
        status[name]=dict(code_cells=sum(c.cell_type=='code' for c in nb.cells),
            executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),errors=len(errors))
        print(name,status[name],flush=True)
    write_json(ROOT/'results/notebook_execution.json',status)
