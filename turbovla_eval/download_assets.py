from huggingface_hub import HfApi, hf_hub_download
from eval_utils import ROOT, ASSETS, write_json
api=HfApi()
repo='H-EmbodVis/TurboVLA'
provenance=ASSETS/'hf_revision.json'
revision=__import__('json').loads(provenance.read_text())['revision'] if provenance.exists() else api.model_info(repo).sha
paths=['checkpoints/libero/turbovla_libero.pth','libero_all4_stats.json','README.md','DINOv3_LICENSE.md','LICENSE']
import shutil
for path in paths:
    destination=ASSETS/path.split('/')[-1]
    if not destination.exists():
        cached=hf_hub_download(repo,path,revision=revision)
        shutil.copy2(cached,destination)
write_json(provenance,dict(repo=repo,revision=revision,files=paths))
