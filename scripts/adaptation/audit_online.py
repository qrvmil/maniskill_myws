"""Verify all saved online rewards/returns and separate teacher/D0 actions."""
import argparse,tarfile
from common import *
from shaping import GAMMA,BETA
def main(label):
 from storage import push_progress
 folder=WORK/'rl'/label;meta=json.loads((folder/'training.json').read_text());size=meta['counters']['environment_steps']
 with np.load(folder/'online_replay.npz',allow_pickle=False) as archive:
  data={k:archive[k] for k in ('actions','base_actions','next_base_actions','rewards','dones','mc_returns')}
 assert len(data['actions'])==size
 with tarfile.open(folder/'collection_traces.tar.gz','r:gz') as archive:
  records=[json.loads(archive.extractfile(member).read()) for member in sorted(archive.getmembers(),key=lambda m:m.name) if member.isfile() and member.name.endswith('.json')]
 offset=warmup=active=successes=0;reward_error=return_error=teacher_error=max_unit=0.
 for record in records:
  evidence=record['rewards'];n=len(evidence);end=offset+n;expected=[]
  for item in evidence:
   assert 0<=item['phi']<=4 and 0<=item['next_phi']<=4
   nxt=0. if item['done'] else item['next_phi']
   expected.append(float(item['success_reward'])+BETA*(GAMMA*nxt-item['phi']))
  expected=np.asarray(expected)
  reward_error=max(reward_error,float(np.max(np.abs(data['rewards'][offset:end]-expected))))
  np.testing.assert_array_equal(data['dones'][offset:end],[int(e['done']) for e in evidence])
  assert evidence[-1]['done'] and not any(e['done'] for e in evidence[:-1])
  np.testing.assert_array_equal(data['next_base_actions'][offset:end-1],data['base_actions'][offset+1:end])
  np.testing.assert_array_equal(data['next_base_actions'][end-1],np.zeros(7,np.float32))
  np.testing.assert_array_equal(data['mc_returns'][offset:end],np.asarray(record['mc_returns'],np.float32))
  running=0.
  for i in range(n-1,-1,-1):
   running=expected[i]+GAMMA*running
   return_error=max(return_error,abs(float(data['mc_returns'][offset+i])-running))
  if record['warmup']:
   warmup+=n;successes+=int(evidence[-1]['success_reward']);teacher=record['teacher_actions'];assert len(teacher)==n
   actions=np.asarray([r['executed_action'] for r in teacher],np.float32)
   bases=np.asarray([r['d0_base_action'] for r in teacher],np.float32)
   wanted=np.asarray([r['teacher_action'] for r in teacher],np.float32)
   unit=np.asarray([r['unit_residual'] for r in teacher],np.float32)
   np.testing.assert_array_equal(data['actions'][offset:end],actions)
   np.testing.assert_array_equal(data['base_actions'][offset:end],bases)
   np.testing.assert_array_equal(np.clip(bases+2*unit,-1,1),actions)
   teacher_error=max(teacher_error,float(np.max(np.abs(actions-wanted))));max_unit=max(max_unit,float(np.max(np.abs(unit))))
  else:active+=n;assert record['teacher_actions'] is None
  offset=end
 assert offset==size and active==meta['counters']['active_steps']
 assert max(reward_error,return_error)<1e-6 and teacher_error<=2e-7 and max_unit<=1
 assert meta['frozen_d0_weights_unchanged']
 proof=dict(passed=True,checkpoint=label,episodes=len(records),transitions=size,warmup_transitions=warmup,active_transitions=active,
  warmup_successes=successes,maximum_reward_error=reward_error,maximum_mc_return_error=return_error,
  maximum_teacher_reconstruction_error=teacher_error,maximum_unit_correction=max_unit,
  teacher_and_d0_cache_actions_separately_verified=True,all_terminal_next_base_actions_zero=True,
  frozen_d0_weights_sha256=meta['base_weights_sha256'],frozen_d0_unchanged=True,final_seeds_used=False)
 write(WORK/'evidence'/('online_audit_'+label+'.json'),proof);print(proof,flush=True)
 push_progress('exp: audit '+label+' online replay rewards, returns and separate teacher/D0 cached actions')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint');main(p.parse_args().checkpoint)
