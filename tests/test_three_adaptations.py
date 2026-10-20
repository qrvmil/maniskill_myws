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

def test_teacher_warmup_stores_executed_teacher_and_advancing_d0_cache():
 from train_residual import TeacherResidual
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy
 from maniskill_myws.pld.libero_runner import run_episode
 class Model:
  def __init__(self,start):self.start=start;self.calls=[]
  def reset(self,seed):self.calls=[]
  def infer(self,obs):
   self.calls.append(int(obs['robot0_eef_pos'][0]))
   return {'actions':np.repeat((self.start+np.arange(5,dtype=np.float32)*.01)[:,None],7,axis=1)}
 class Env:
  prompt='test';horizon=7
  def observation(self):
   im=np.zeros((4,4,3),np.uint8)
   return dict(agentview_image=im,robot0_eye_in_hand_image=im,robot0_eef_pos=np.array([self.i,0,0],np.float32),
     robot0_eef_quat=np.array([0,0,0,1],np.float32),robot0_gripper_qpos=np.zeros(2,np.float32))
  def reset(self,*,seed):
   self.i=0;self.raw=self.observation()
   return self.raw,dict(reset_hash='fixed',initial_state=np.array([0.]),sampled_initial_state=np.array([0.]))
  def step(self,action):
   self.i+=1;self.raw=self.observation();done=self.i==7
   return self.raw,float(done),done,False,dict(success=done)
  def physics_state(self):return np.array([float(self.i)])
 env=Env();d0_model=Model(-.2);teacher_model=Model(.6)
 d0=ChunkedBasePolicy(d0_model,replan_steps=5);teacher=ChunkedBasePolicy(teacher_model,replan_steps=5)
 residual=TeacherResidual(teacher,env);residual.reset(42)
 row,tr=run_episode(env,d0,seed=42,image_size=4,residual=residual,residual_scale=2)
 assert row['success'] and row['length']==7
 assert d0_model.calls==teacher_model.calls==[0,5]
 for i,t in enumerate(tr):
  np.testing.assert_allclose(t['base_action'],np.full(7,-.2+(i%5)*.01),atol=1e-7)
  np.testing.assert_allclose(t['action'],np.full(7,.6+(i%5)*.01),atol=1e-7)
  np.testing.assert_array_equal(t['next_base_action'],tr[i+1]['base_action'] if i+1<len(tr) else np.zeros(7))
 assert max(r['reconstruction_max_error'] for r in residual.records)<2e-7
