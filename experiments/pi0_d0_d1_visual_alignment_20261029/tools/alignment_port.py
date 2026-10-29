"""PyTorch port of pinned spatial alignment; original Pi0 scopes are preserved."""
import torch
import torch.nn.functional as F

def project(params,hidden,freeze=True):
 p={k:v.detach() if freeze else v for k,v in params.items()}
 x=hidden.float();mean=x.mean(-1,keepdim=True)
 x=(x-mean)*torch.rsqrt((x-mean).square().mean(-1,keepdim=True)+1e-5)
 x=x*p['scale']+p['bias']
 return F.gelu(x@p['w1']+p['b1'],approximate='tanh')@p['w2']+p['b2']

def cosine_loss(student,teacher,mask):
 if student.shape!=teacher.shape or student.ndim!=5 or student.shape[1:4]!=(2,16,16):
  raise ValueError('Student/teacher spatial/view grids must match exactly')
 if mask.shape!=student.shape[:2]:raise ValueError('View mask shape mismatch')
 s=F.normalize(student.float(),dim=-1,eps=1e-8)
 t=F.normalize(teacher.detach().float(),dim=-1,eps=1e-8)
 return ((1-(s*t).sum(-1))*mask[:,:,None,None]).sum()/(mask.sum()*256)

def _prefix_layer(layer,llm,hidden,attention_mask,position_ids):
 # The exact pinned joint-forward layer arithmetic, restricted to its prefix.
 from transformers.models.gemma import modeling_gemma as g
 norm,gate=layer.input_layernorm(hidden,cond=None)
 shape=(*norm.shape[:-1],-1,layer.self_attn.head_dim)
 q=layer.self_attn.q_proj(norm).view(shape).transpose(1,2)
 k=layer.self_attn.k_proj(norm).view(shape).transpose(1,2)
 v=layer.self_attn.v_proj(norm).view(shape).transpose(1,2)
 dummy=torch.zeros(q.shape[0],q.shape[2],q.shape[-1],device=q.device,dtype=q.dtype)
 cos,sin=llm.rotary_emb(dummy,position_ids)
 q,k=g.apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1)
 out,_=g.eager_attention_forward(layer.self_attn,q,k,v,attention_mask,layer.self_attn.scaling)
 out=out.reshape(q.shape[0],-1,q.shape[1]*q.shape[-1])
 if out.dtype!=layer.self_attn.o_proj.weight.dtype:out=out.to(layer.self_attn.o_proj.weight.dtype)
 hidden=g._gated_residual(hidden,layer.self_attn.o_proj(out),gate)
 norm,gate=layer.post_attention_layernorm(hidden,cond=None)
 if layer.mlp.up_proj.weight.dtype==torch.bfloat16:norm=norm.to(torch.bfloat16)
 return g._gated_residual(hidden,layer.mlp(norm),gate)

def middle_state(model,observation,depth=9):
 from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
 names=list(observation.images)
 if names!=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']:
  raise ValueError('Unexpected camera/patch order')
 images,masks,tokens,token_masks,_=model._preprocess_observation(observation,train=False)
 hidden,pad,ar=model.embed_prefix(images,masks,tokens,token_masks)
 llm=model.paligemma_with_expert.paligemma.language_model
 if llm.layers[0].self_attn.q_proj.weight.dtype==torch.bfloat16:hidden=hidden.to(torch.bfloat16)
 att=model._prepare_attention_masks_4d(make_att_2d_masks(pad,ar));positions=pad.cumsum(-1)-1
 for layer in llm.layers[:depth]:
  hidden=model._apply_checkpoint(lambda h,layer=layer:_prefix_layer(layer,llm,h,att,positions),hidden)
 hidden=hidden[:,:3*256].reshape(hidden.shape[0],3,16,16,-1)
 return hidden[:,:2],torch.stack([observation.image_masks[n] for n in names[:2]],dim=1)

def disable_extra_augmentation(model):
 import types
 original=model._preprocess_observation
 def preprocess(self,observation,*,train=True):return original(observation,train=False)
 model._preprocess_observation=types.MethodType(preprocess,model)
