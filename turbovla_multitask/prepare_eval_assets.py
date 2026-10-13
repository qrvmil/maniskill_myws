"""Restore encoder folders from OUR full EMA policy; download only BERT tokenizer."""
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from prepare import sha256,write_json


def prepare_policy_assets(checkpoint,root):
    import torch
    from transformers import BertConfig,BertModel,DINOv3ViTConfig,DINOv3ViTModel
    from huggingface_hub import hf_hub_download
    import shutil
    checkpoint=Path(checkpoint);root=Path(root)
    digest=sha256(checkpoint)
    directory=root/digest
    marker=directory/'preparation.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['checkpoint_sha256']!=digest or record['weights']!='ema_model_state_dict':
            raise ValueError('Cached evaluation encoder identity differs')
        for name,expected in record['files'].items():
            if sha256(directory/name)!=expected: raise ValueError(f'Cached evaluation asset differs: {name}')
        return directory/'dinov3',directory/'bert'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('experiment',{}).get('smoke_steps') is not None or payload.get('step')!=3001:
        raise ValueError('Restore evaluation assets only from a full trained experiment checkpoint')
    state=payload['ema_model_state_dict']
    prefixes={'dinov3':'vision_encoder.backbone.','bert':'text_encoder.bert.'}
    counts={}
    for name,prefix in prefixes.items():
        part={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
        if not part: raise ValueError(f'Missing full {name} weights')
        if name=='dinov3':
            if part['embeddings.cls_token'].shape!=(1,1,768) or part['embeddings.register_tokens'].shape!=(1,4,768):
                raise ValueError('Expected original TurboVLA LIBERO ViT-B/16 architecture')
            config=DINOv3ViTConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
                num_attention_heads=12,num_register_tokens=4,patch_size=16,rope_theta=100.,layer_norm_eps=1e-5,layerscale_value=1.)
            model=DINOv3ViTModel(config)
        else:
            model=BertModel(BertConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,num_attention_heads=12))
        model.load_state_dict(part,strict=True)
        destination=directory/name
        destination.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(destination,safe_serialization=True)
        counts[name]=len(part)
        del model,part
        if name=='dinov3':
            write_json(destination/'preprocessor_config.json',dict(image_processor_type='DINOv3ViTImageProcessorFast',
                do_resize=False,do_center_crop=False,image_mean=[.485,.456,.406],image_std=[.229,.224,.225],rescale_factor=1/255))
        else:
            for filename in ['vocab.txt','tokenizer.json','tokenizer_config.json']:
                cached=hf_hub_download('google-bert/bert-base-uncased',filename,
                    revision='86b5e0934494bd15c9632b12f734a8a67f723594')
                shutil.copy2(cached,destination/filename)
    files={str(path.relative_to(directory)):sha256(path) for path in directory.rglob('*') if path.is_file()}
    write_json(marker,dict(checkpoint_sha256=digest,weights='ema_model_state_dict',strict_loads=counts,
                          files=files,source='Full trained policy; no released LIBERO policy or original encoder weights downloaded'))
    return directory/'dinov3',directory/'bert'
