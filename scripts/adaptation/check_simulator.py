"""Real simulator/reset/shaping/video smoke before any long optimizer run."""
from common import *
def main():
 from evaluate import PairedVideoEnv
 from shaping import ShapedEnv
 from maniskill_myws.pld.libero_progress import task_progress
 rows=[]
 for task in TASKS:
  env=ShapedEnv(PairedVideoEnv(task,'simulator_smoke'))
  raw,first=env.reset(seed=23020)
  initial_progress=task_progress(env,raw)
  a=np.array([0.]*6+[-1.],np.float32)
  for _ in range(3):raw,reward,terminated,truncated,info=env.step(a)
  assert len(env.frames)==4
  rewards=list(env.reward_evidence);assert all(np.isfinite(r['reward']) for r in rewards)
  raw,second=env.reset(seed=23020)
  assert first['reset_hash']==second['reset_hash']
  np.testing.assert_array_equal(first['initial_state'],second['initial_state'])
  rows.append(dict(task=task,reset_hash=first['reset_hash'],initial_progress=initial_progress,shaping_rewards=rewards))
  env.close()
 write(WORK/'evidence/simulator_smoke.json',dict(passed=True,episodes=rows))
 print('REAL_SIMULATOR_PAIRED_RESETS_AND_SHAPING_PASS',flush=True)
if __name__=='__main__':main()
