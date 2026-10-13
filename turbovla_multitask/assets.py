"""Download original, not LIBERO-trained, initialization assets."""
import argparse
import os
import sys
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
from data import WORK, ROOT
from prepare import write_json, sha256

ASSET_REVISIONS = {
    'dinov3':('facebook/dinov3-vitb16-pretrain-lvd1689m','5931719e67bbdb9737e363e781fb0c67687896bc'),
    'bert':('google-bert/bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594'),
}
GROUNDING_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
MIRROR_REPO='nyu-visionx/RAEv2-models'
MIRROR_REV='9770b7b980fa1875c8e6d65f226c615c0ce908a8'
MIRROR_FILE='encoders/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
MIRROR_HASH='73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'
META_REV='6876159a11b4df116f30f667f8c9888617df0751'


def prepare_original_mirror():
    """User-authorized public mirror; verify bytes, strict loads and forward parity."""
    import subprocess
    import torch
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import DINOv3ViTConfig,DINOv3ViTModel
    from convert_dinov3 import convert_state_dict
    source_path=Path(hf_hub_download(MIRROR_REPO,MIRROR_FILE,revision=MIRROR_REV,
                                   local_dir=WORK/'assets'/'original_mirror'))
    if sha256(source_path)!=MIRROR_HASH:
        raise ValueError('Original DINOv3 mirror checksum differs')
    meta_root=ROOT/'third_party'/'dinov3'
    if not (meta_root/'.git').exists():
        subprocess.run(['git','clone','https://github.com/facebookresearch/dinov3.git',str(meta_root)],check=True)
    subprocess.run(['git','-C',str(meta_root),'checkout',META_REV],check=True)
    sys.path.insert(0,str(meta_root))
    from dinov3.hub.backbones import dinov3_vitb16
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    config=DINOv3ViTConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
        num_attention_heads=12,num_register_tokens=4,patch_size=16,rope_theta=100.,
        layer_norm_eps=1e-5,layerscale_value=1.)
    model=DINOv3ViTModel(config)
    model.load_state_dict(convert_state_dict(source),strict=True)
    reference=dinov3_vitb16(pretrained=False)
    reference.load_state_dict(source,strict=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model.eval().to(device);reference.eval().to(device)
    x=torch.randn(1,3,256,256,device=device,generator=torch.Generator(device=device).manual_seed(42))
    with torch.no_grad():
        original=reference.forward_features(x)['x_norm_patchtokens']
        converted=model(x).last_hidden_state[:,5:]
    default_error=(original-converted).abs().max().item()
    cosine=torch.nn.functional.cosine_similarity(original.flatten(),converted.flatten(),dim=0).item()
    if cosine<.99999 or default_error>.01:
        raise ValueError(f'Original/converted DINOv3 forward discrepancy: {default_error}, {cosine}')
    # Meta saved its deterministic periods in BF16. Isolate that rounding from
    # learned-weight mapping: with identical RoPE both implementations agree.
    reference.rope_embed.periods.data=1/model.rope_embeddings.inv_freq
    with torch.no_grad(): original=reference.forward_features(x)['x_norm_patchtokens']
    torch.testing.assert_close(original,converted,rtol=2e-4,atol=2e-4)
    same_rope_error=(original-converted).abs().max().item()
    path=WORK/'assets'/'dinov3'
    model.cpu().save_pretrained(path)
    write_json(path/'preprocessor_config.json',dict(image_processor_type='DINOv3ViTImageProcessorFast',
        do_resize=False,do_center_crop=False,image_mean=[.485,.456,.406],image_std=[.229,.224,.225],rescale_factor=1/255))
    import shutil
    shutil.copy2(meta_root/'LICENSE.md',path/'LICENSE.md')
    provenance=dict(original_model=ASSET_REVISIONS['dinov3'][0],mirror_repo=MIRROR_REPO,
        mirror_revision=MIRROR_REV,mirror_file=MIRROR_FILE,source_sha256=MIRROR_HASH,
        meta_code_revision=META_REV,source_tensors=len(source),target_tensors=len(model.state_dict()),
        strict_meta=True,strict_transformers=True,default_rope_max_abs_error=default_error,
        default_rope_cosine=cosine,same_rope_max_abs_error=same_rope_error,
        difference='Meta checkpoint rounds deterministic RoPE periods to BF16; Transformers regenerates FP32 periods',
        no_libero_policy_weights=True)
    write_json(WORK/'assets'/'dinov3_conversion.json',provenance)
    print('ORIGINAL_MIRROR_VERIFIED',provenance,flush=True)
    return provenance


def prepare_assets():
    from huggingface_hub import snapshot_download
    import urllib.request
    manifest = {}
    for name,(repo,revision) in ASSET_REVISIONS.items():
        path = WORK/'assets'/name
        if name=='dinov3':
            provenance=prepare_original_mirror()
        else:
            snapshot_download(repo,revision=revision,local_dir=path,
                allow_patterns=['config.json','model.safetensors','preprocessor_config.json','tokenizer.json','tokenizer_config.json','vocab.txt'])
        if not (path/'model.safetensors').exists():
            raise FileNotFoundError(f'Missing pretrained weights: {path}')
        manifest[name] = dict(repo=MIRROR_REPO if name=='dinov3' else repo,
            revision=MIRROR_REV if name=='dinov3' else revision,
            files={p.name:sha256(p) for p in path.iterdir() if p.is_file()})
        if name=='dinov3': manifest[name]['conversion']=provenance
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
