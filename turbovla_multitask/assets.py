"""Download original, not LIBERO-trained, initialization assets."""
import argparse
import os
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
from data import WORK
from prepare import write_json, sha256

ASSET_REVISIONS = {
    'dinov3':('facebook/dinov3-vitb16-pretrain-lvd1689m','5931719e67bbdb9737e363e781fb0c67687896bc'),
    'bert':('google-bert/bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594'),
}
GROUNDING_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'


def prepare_assets():
    from huggingface_hub import snapshot_download
    import urllib.request
    manifest = {}
    for name,(repo,revision) in ASSET_REVISIONS.items():
        path = WORK/'assets'/name
        snapshot_download(repo,revision=revision,local_dir=path,
            allow_patterns=['config.json','model.safetensors','preprocessor_config.json','tokenizer.json','tokenizer_config.json','vocab.txt'])
        if not (path/'model.safetensors').exists():
            raise FileNotFoundError(f'Missing pretrained weights: {path}')
        manifest[name] = dict(repo=repo,revision=revision,
            files={p.name:sha256(p) for p in path.iterdir() if p.is_file()})
    path = WORK/'assets'/'groundingdino_swint_ogc.pth'
    if not path.exists():
        tmp = path.with_suffix('.download')
        urllib.request.urlretrieve(GROUNDING_URL,tmp)
        tmp.replace(path)
    manifest['groundingdino'] = dict(url=GROUNDING_URL,sha256=sha256(path))
    write_json(WORK/'assets'/'manifest.json',manifest)
    print('ORIGINAL_ASSETS_VERIFIED',flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare_assets()
