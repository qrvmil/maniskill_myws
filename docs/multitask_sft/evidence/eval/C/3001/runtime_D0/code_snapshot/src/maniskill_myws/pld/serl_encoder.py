"""PyTorch port of SERL ResNetV1-10 (GN4), with exact JAX SAME padding.

Reference: rail-berkeley/serl, revision 1fa2af7496be042e43d36120a7c8d1b71f1bfb59.
Weights are the official ImageNet-1K release, NOT a LIBERO-trained model.
Frozen per-view trunk and trainable spatial pooling follow SERL. No dropout
in this port (PLD Table 5 specifies Q dropout 0); no random crop yet.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def same_pad(x,kernel,stride,value=0.):
    h,w=x.shape[-2:]
    ph=max((math.ceil(h/stride)-1)*stride+kernel-h,0)
    pw=max((math.ceil(w/stride)-1)*stride+kernel-w,0)
    return F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2),value=value)


class Block(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__()
        self.stride=stride
        self.conv1=nn.Conv2d(cin,cout,3,stride=stride,bias=False)
        self.norm1=nn.GroupNorm(4,cout,eps=1e-5)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.norm2=nn.GroupNorm(4,cout,eps=1e-5)
        self.proj=nn.Conv2d(cin,cout,1,stride=stride,bias=False) if cin!=cout else None
        self.norm_proj=nn.GroupNorm(4,cout,eps=1e-5) if self.proj is not None else None

    def forward(self,x):
        y=F.relu(self.norm1(self.conv1(same_pad(x,3,self.stride))))
        y=self.norm2(self.conv2(y))
        shortcut=x if self.proj is None else self.norm_proj(self.proj(x))
        return F.relu(y+shortcut)


class SERLTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_init=nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
        self.norm_init=nn.GroupNorm(4,64,eps=1e-5)
        self.blocks=nn.Sequential(Block(64,64,1),Block(64,128,2),Block(128,256,2),Block(256,512,2))

    def forward(self,x):
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        x=F.relu(self.norm_init(self.conv_init((x-mean)/std)))
        x=F.max_pool2d(same_pad(x,3,2,float('-inf')),3,2)
        return self.blocks(x)


class SpatialHead(nn.Module):
    def __init__(self,height,width,latent_dim):
        super().__init__()
        self.kernel=nn.Parameter(torch.empty(height,width,512,8))
        nn.init.normal_(self.kernel,std=1/math.sqrt(height*width*512))
        self.projection=nn.Sequential(nn.Linear(512*8,latent_dim),nn.LayerNorm(latent_dim,eps=1e-6),nn.Tanh())

    def forward(self,x):
        features=torch.einsum('bchw,hwcf->bcf',x,self.kernel).flatten(1)
        return self.projection(features)


class SERLResNet10Encoder(nn.Module):
    def __init__(self,image_shape,latent_dim):
        super().__init__()
        views,h,w,c=image_shape
        if c!=3:raise ValueError('SERL requires RGB images per camera')
        self.views=views
        self.trunk=SERLTrunk().requires_grad_(False)
        self.heads=nn.ModuleList([SpatialHead(math.ceil(h/32),math.ceil(w/32),latent_dim) for _ in range(views)])

    def forward(self,images):
        b,_,h,w=images.shape
        with torch.no_grad():
            features=self.trunk(images.reshape(b*self.views,3,h,w)).reshape(b,self.views,512,math.ceil(h/32),math.ceil(w/32))
        return torch.cat([head(features[:,i]) for i,head in enumerate(self.heads)],dim=-1)

    def load_pretrained(self,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state=payload.get('trunk',{})
        expected=self.trunk.state_dict()
        if set(state)!=set(expected) or any(state[k].shape!=v.shape for k,v in expected.items()):
            raise ValueError('Pretrained SERL trunk keys/shapes do not match completely')
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite pretrained trunk')
        self.trunk.load_state_dict(state,strict=True)
        return len(state)


def convert_serl_params(params):
    """Convert official Flax HWIO kernels/GN affine tensors, excluding classifier."""
    state={}
    def conv(key,p):state[key+'.weight']=torch.from_numpy(np.array(p['kernel']).transpose(3,2,0,1).copy())
    def norm(key,p):
        state[key+'.weight']=torch.from_numpy(np.array(p['scale']).copy())
        state[key+'.bias']=torch.from_numpy(np.array(p['bias']).copy())
    conv('conv_init',params['conv_init']);norm('norm_init',params['norm_init'])
    for i in range(4):
        p=params[f'ResNetBlock_{i}'];prefix=f'blocks.{i}'
        conv(prefix+'.conv1',p['Conv_0']);conv(prefix+'.conv2',p['Conv_1'])
        norm(prefix+'.norm1',p['MyGroupNorm_0']);norm(prefix+'.norm2',p['MyGroupNorm_1'])
        if i:
            conv(prefix+'.proj',p['conv_proj']);norm(prefix+'.norm_proj',p['norm_proj'])
    return state
