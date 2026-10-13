"""Publish full final model+stats+provenance, then verify remote file hashes."""
import argparse
import os
import shutil
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
from data import WORK, HF_REPO, TRAIN_SETS
from prepare import write_json,sha256
from artifacts import verified_final


def publish(variant):
    from huggingface_hub import HfApi,hf_hub_download
    checkpoint,stats,metadata = verified_final(variant)
    staging = WORK/variant/'publish'
    staging.mkdir(parents=True,exist_ok=True)
    files = {
        'final.pth':checkpoint,'stats.json':stats,
        'training_config.json':WORK/variant/'train'/'config.json',
        'training_complete.json':WORK/variant/'train'/'complete.json',
        'train.jsonl':WORK/variant/'train'/'train.jsonl',
        'data_manifest.json':WORK/'data_manifest.json',
        'initialization.json':WORK/'initialization.json',
        'assets_manifest.json':WORK/'assets'/'manifest.json',
    }
    for name,path in files.items():
        shutil.copy2(path,staging/name)
    hashes = {name:sha256(staging/name) for name in files}
    write_json(staging/'file_hashes.json',hashes)
    (staging/'README.md').write_text(
        f'# TurboVLA variant {variant}\n\nTraining tasks: {TRAIN_SETS[variant]}. '
        'Full raw and EMA policy weights; primary evaluation uses EMA. '
        'Original pretrained backbones; no released LIBERO policy initialization. '
        'DINOv3 license also applies to the contained visual backbone. '
        'Load with the pinned TurboVLAPolicy and the bundled stats.json (stats_key=train).\n')
    api = HfApi()
    api.create_repo(HF_REPO,private=True,exist_ok=True)
    info = api.upload_folder(repo_id=HF_REPO,folder_path=staging,path_in_repo=variant,
                             commit_message=f'Publish independently trained TurboVLA {variant} and provenance')
    for name,expected in hashes.items():
        cached = hf_hub_download(HF_REPO,f'{variant}/{name}',revision=info.oid)
        if sha256(cached) != expected:
            raise ValueError(f'Remote artifact mismatch: {variant}/{name}')
    record = dict(repo=HF_REPO,revision=info.oid,variant=variant,hashes=hashes,
                  checkpoint_sha256=metadata['checkpoint_sha256'],verified_download=True)
    write_json(WORK/variant/'hf_publication.json',record)
    print('REMOTE_WEIGHTS_VERIFIED',record,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=list(TRAIN_SETS),required=True)
    publish(parser.parse_args().variant)
