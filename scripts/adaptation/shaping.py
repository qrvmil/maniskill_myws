"""One bounded Markov potential/reward/return definition for offline and online data."""
import numpy as np
GAMMA=.99
BETA=.1

def potential(progress,reference_height):
 reach=1-np.tanh(10*progress['reach_distance'])
 grasp=float(progress['grasp'])
 lift=float(np.clip((progress['bowl_height']-reference_height)/.08,0,1))
 transport=1-np.tanh(10*progress['bowl_plate_distance'])
 return float(reach+grasp+lift*(1+transport))

def reward(success,phi,next_phi,done):
 # Both genuine success and finite-horizon truncation enter an absorbing state.
 return float(success)+BETA*(GAMMA*(0. if done else float(next_phi))-float(phi))

def returns(transitions):
 out=np.zeros(len(transitions),np.float32);running=0.
 for i in range(len(transitions)-1,-1,-1):
  t=transitions[i];running=t['reward']+GAMMA*(1-float(t['done']))*running;out[i]=running
 return out

class ShapedEnv:
 """Delegates simulator success unchanged and records shaping evidence at every boundary."""
 def __init__(self,env):self._env=env;self.potentials=[];self.reward_evidence=[]
 def __getattr__(self,name):return getattr(self._env,name)
 def reset(self,*,seed):
  from maniskill_myws.pld.libero_progress import task_progress
  raw,info=self._env.reset(seed=seed);p=task_progress(self._env,raw)
  self.reference_height=p['bowl_height'];self.phi=potential(p,self.reference_height)
  self.potentials=[self.phi];self.reward_evidence=[];self.raw=raw
  return raw,info
 def step(self,action):
  from maniskill_myws.pld.libero_progress import task_progress
  raw,success,terminated,truncated,info=self._env.step(action);done=terminated or truncated
  nxt=potential(task_progress(self._env,raw),self.reference_height)
  shaped=reward(success,self.phi,nxt,done)
  self.reward_evidence.append(dict(success_reward=success,phi=self.phi,next_phi=nxt,absorbing_next_phi=0. if done else nxt,done=bool(done),reward=shaped))
  self.phi=nxt;self.potentials.append(nxt);self.raw=raw
  return raw,shaped,terminated,truncated,info
 def close(self):self._env.close()
