"""Lossless learned-tensor mapping from Meta ViT-B/16 to Transformers4.56.2."""
import re
import torch

TOP_KEYS={'cls_token':'embeddings.cls_token','storage_tokens':'embeddings.register_tokens',
          'mask_token':'embeddings.mask_token','patch_embed.proj.weight':'embeddings.patch_embeddings.weight',
          'patch_embed.proj.bias':'embeddings.patch_embeddings.bias','norm.weight':'norm.weight','norm.bias':'norm.bias'}
BLOCK_KEYS={'norm1':'norm1','norm2':'norm2','attn.proj':'attention.o_proj',
            'mlp.fc1':'mlp.up_proj','mlp.fc2':'mlp.down_proj',
            'ls1.gamma':'layer_scale1.lambda1','ls2.gamma':'layer_scale2.lambda1'}


def convert_state_dict(source):
    converted={}
    for name,value in source.items():
        if name=='rope_embed.periods':
            # Deterministic RoPE periods are a non-persistent buffer in Transformers.
            # They are generated from rope_theta; parity is checked by the asset script.
            continue
        if name in TOP_KEYS:
            converted[TOP_KEYS[name]]=value.unsqueeze(1) if name=='mask_token' else value
            continue
        match=re.fullmatch(r'blocks\.(\d+)\.(.*)',name)
        if not match: raise ValueError(f'Unknown source tensor: {name}')
        index,suffix=match.groups();prefix=f'layer.{index}.'
        if suffix=='attn.qkv.bias_mask': continue
        if suffix in ('attn.qkv.weight','attn.qkv.bias'):
            kind=suffix.rsplit('.',1)[1]
            if value.shape[0]%3: raise ValueError(f'Invalid QKV shape: {name}')
            if kind=='bias': value=value*source.get(f'blocks.{index}.attn.qkv.bias_mask',torch.ones_like(value))
            parts=value.chunk(3,dim=0)
            for projection,part in zip(('q','k','v'),parts):
                if projection=='k' and kind=='bias':
                    if torch.count_nonzero(part): raise ValueError('Nonzero unmasked key bias cannot be discarded')
                    continue
                converted[prefix+f'attention.{projection}_proj.{kind}']=part.contiguous()
            continue
        mapped=next((target+suffix[len(original):] for original,target in BLOCK_KEYS.items()
                     if suffix==original or suffix.startswith(original+'.')),None)
        if mapped is None: raise ValueError(f'Unknown source tensor: {name}')
        converted[prefix+mapped]=value
    return converted
