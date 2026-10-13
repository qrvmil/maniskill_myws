import json
import tempfile
import unittest
from pathlib import Path
import nbformat
from data import ROOT
from prepare import sha256


class NotebookContracts(unittest.TestCase):
    def test_local_verified_weights_do_not_require_a_publication_record(self):
        notebook=nbformat.read(ROOT/'notebooks/02_turbovla_multitask_eval.ipynb',as_version=4)
        source=next(c.source for c in notebook.cells if c.cell_type=='code' and 'from artifacts import require_final' in c.source)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';(work/'A/train').mkdir(parents=True)
            checkpoint=work/'A/train/final.pth';checkpoint.write_bytes(b'unit-fixture')
            stats=work/'A/stats.json';stats.write_text('{}')
            metadata=dict(variant='A',step=3001,training_tasks=['D0'],smoke_only=False,
                primary_weights='ema_model_state_dict',checkpoint_sha256=sha256(checkpoint),stats_sha256=sha256(stats))
            (work/'A/train/complete.json').write_text(json.dumps(metadata))
            context=dict(REPO=root,WORK=work,VARIANT='A',DOWNLOAD_FROM_HF=False,
                         Path=Path,json=json,sha256=sha256,HF_REPO='unused')
            exec(source,context)
            self.assertEqual(context['checkpoint'],checkpoint)
            self.assertIsNone(context['revision'])
            self.assertEqual(context['metadata'],metadata)


if __name__=='__main__':unittest.main()
