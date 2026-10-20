"""Fresh versus restored reset must refresh identical hidden sensor caches."""
import numpy as np
from maniskill_myws.pld.libero_backend import LiberoEnv

def test_fresh_and_explicit_states_follow_identical_cache_refresh():
 class Simulator:
  def seed(self,seed):pass
  def reset(self):self.state=np.array([0.,1.]);self.cache=0;return self.cache
  def get_sim_state(self):return self.state
  def set_init_state(self,state):self.state=np.array(state).copy();self.cache=1;return self.cache
  def step(self,action):self.state[0]+=1;return self.cache,0,False,{}
 env=LiberoEnv.__new__(LiberoEnv);env.env=Simulator();env.settle_steps=2;env.task_id=0
 env.task=type('Task',(),{'name':'test'})()
 fresh,info=env.reset(seed=1)
 restored,other=env.reset(seed=1,initial_state=info['sampled_initial_state'])
 assert fresh==restored==1
 assert info['reset_hash']==other['reset_hash']
 np.testing.assert_array_equal(info['initial_state'],other['initial_state'])
