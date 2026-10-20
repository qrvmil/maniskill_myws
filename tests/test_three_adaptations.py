"""Scientific invariants unique to the three-method experiment."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/adaptation'))
from shaping import reward,returns,GAMMA,BETA
from maniskill_myws.pld.libero_backend import ActionContract

def test_teacher_action_reachable_in_controller_units():
 rng=np.random.default_rng(1);base=rng.uniform(-1,1,(100,7));teacher=rng.uniform(-1,1,(100,7))
 base[0,6]=-1;teacher[0,6]=1
 for b,t in zip(base,teacher):
  delta=(t-b)/2
  assert np.max(np.abs(delta))<=1
  np.testing.assert_allclose(ActionContract(2).compose(b,delta),t,atol=2e-7)

def test_shaping_telescopes_with_absorbing_terminal():
 phi=[.4,1.2,.9,2.3];success=[0,0,1]
 ts=[dict(reward=reward(s,phi[i],phi[i+1],i==2),done=i==2) for i,s in enumerate(success)]
 target=GAMMA**2-BETA*phi[0]
 assert returns(ts)[0]==pytest.approx(target)
 # Absorbing boundary must ignore the simulator's nonzero terminal potential.
 assert reward(1,phi[-2],100,True)==reward(1,phi[-2],0,True)

def test_zero_residual_is_bitwise_base_action():
 rng=np.random.default_rng(7)
 for b in rng.uniform(-1,1,(50,7)).astype(np.float32):
  np.testing.assert_array_equal(ActionContract(2).compose(b,np.zeros(7,np.float32)),b)
