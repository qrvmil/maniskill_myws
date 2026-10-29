"""Check the port against the pinned JAX source, including frozen gradients/masks."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
BASE=Path('/workspace/visual-alignment-repro-20261029')
sys.path.insert(0,str(BASE/'pi0-d0d1-visual-alignment'))
from alignment.model import project as reference_project,cosine_loss as reference_cosine
import jax
import jax.numpy as jnp
from alignment_port import project,cosine_loss

def test_projector_and_masked_cosine_match_source_and_student_gradients():
 rng=np.random.default_rng(7)
 arrays=dict(scale=np.ones(8,dtype='f'),bias=np.zeros(8,dtype='f'),
 w1=rng.normal(size=(8,6)).astype('f')*.1,b1=np.zeros(6,dtype='f'),
 w2=rng.normal(size=(6,4)).astype('f')*.1,b2=np.zeros(4,dtype='f'))
 p={k:torch.tensor(v,requires_grad=True) for k,v in arrays.items()}
 x=rng.normal(size=(2,2,16,16,8)).astype('f');t=rng.normal(size=(2,2,16,16,4)).astype('f')
 mask=np.array([[True,False],[True,True]])
 h=torch.tensor(x,requires_grad=True);teacher=torch.tensor(t,requires_grad=True)
 y=project(p,h);l=cosine_loss(y,teacher,torch.tensor(mask))
 ref=reference_project({k:jnp.array(v) for k,v in arrays.items()},jnp.array(x))
 np.testing.assert_allclose(y.detach(),np.asarray(ref),rtol=2e-5,atol=2e-6)
 np.testing.assert_allclose(l.detach(),np.asarray(reference_cosine(ref,jnp.array(t),jnp.array(mask))),rtol=2e-6)
 refgrad=jax.grad(lambda v:reference_cosine(reference_project({k:jnp.array(a) for k,a in arrays.items()},v),jnp.array(t),jnp.array(mask)))(jnp.array(x))
 l.backward();np.testing.assert_allclose(h.grad,np.asarray(refgrad),rtol=2e-4,atol=2e-7)
 assert all(v.grad is None for v in p.values()) and teacher.grad is None
 assert h.grad[0,1].count_nonzero()==0
 assert h.grad[0,0].count_nonzero()>0
 altered=teacher.detach().clone();altered[0,1]=1e6
 assert torch.equal(l.detach(),cosine_loss(y.detach(),altered,torch.tensor(mask)))
 assert torch.isnan(cosine_loss(y.detach(),teacher.detach(),torch.zeros((2,2),dtype=torch.bool)))

def test_spatial_view_shape_cannot_silently_broadcast():
 with pytest.raises(ValueError):cosine_loss(torch.ones(1,2,16,16,4),torch.ones(1,1,16,16,4),torch.ones(1,2,dtype=torch.bool))
 with pytest.raises(ValueError):cosine_loss(torch.ones(1,2,16,16,4),torch.ones(1,2,16,16,4),torch.ones(1,3,dtype=torch.bool))
