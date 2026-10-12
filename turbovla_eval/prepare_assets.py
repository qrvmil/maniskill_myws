"""Extract already-released backbone weights; do not download gated DINOv3 weights."""
import json
import os
from pathlib import Path
import urllib.request
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
from eval_utils import ASSETS, ROOT, TURBO_ROOT, file_hash, write_json


def prepare_assets():
    import torch
    from transformers import BertConfig, BertModel, DINOv3ViTConfig, DINOv3ViTModel
    checkpoint = torch.load(ASSETS/'turbovla_libero.pth', map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    config = checkpoint['model_config']
    print('Released model config:',json.dumps(config,indent=2))
    prefixes={'dinov3':'vision_encoder.backbone.','bert':'text_encoder.bert.'}
    counts={}
    for name,prefix in prefixes.items():
        directory=ASSETS/name
        directory.mkdir(exist_ok=True)
        part={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if not part: raise ValueError(f'No complete {name} weights in checkpoint')
        if name=='dinov3':
            # Official DINOv3 ViT-B/16 architecture. Sizes checked from released tensors.
            hidden=part['embeddings.cls_token'].shape[-1]
            registers=part['embeddings.register_tokens'].shape[1]
            layers=len({k.split('.')[1] for k in part if k.startswith('layer.')})
            intermediate=part['layer.0.mlp.up_proj.weight'].shape[0]
            architecture=DINOv3ViTConfig(hidden_size=hidden, intermediate_size=intermediate,
                num_hidden_layers=layers, num_attention_heads=12, num_register_tokens=registers,
                patch_size=16, rope_theta=100.0, layer_norm_eps=1e-5, layerscale_value=1e-5)
            model=DINOv3ViTModel(architecture)
            preprocessing=dict(image_mean=[.485,.456,.406],image_std=[.229,.224,.225],rescale_factor=1/255)
            write_json(directory/'preprocessor_config.json',preprocessing)
        else:
            architecture=BertConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,num_attention_heads=12)
            model=BertModel(architecture)
            for filename in ['config.json','vocab.txt','tokenizer.json','tokenizer_config.json']:
                if not (directory/filename).exists():
                    urllib.request.urlretrieve(f'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/{filename}',directory/filename)
        model.load_state_dict(part,strict=True)
        model.save_pretrained(directory,safe_serialization=True)
        counts[name]=dict(tensors=len(part),parameters=sum(v.numel() for v in model.parameters()),strict_load=True)
        del model,part
    import shutil
    shutil.copy2(TURBO_ROOT/'experiments/libero/configs/libero_all4_stats.json',ASSETS/'libero_all4_stats.json')
    write_json(ASSETS/'preparation.json',dict(checkpoint_sha256=file_hash(ASSETS/'turbovla_libero.pth'),
        backbone_source='Exact tensor subsets from public H-EmbodVis/TurboVLA checkpoint',
        dinov3_architecture_source='facebookresearch/dinov3 dinov3_vitb16 + transformers 4.56.2',
        dinov3_preprocessing_source='TurboVLA manual processor defaults: ImageNet RGB mean/std; no resize/crop',
        backbones=counts,model_config=config))

if __name__=='__main__': prepare_assets()
