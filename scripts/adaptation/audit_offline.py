"""Read-only audit of all native D1 replay actions/images, rewards and returns."""
from common import *
from shaping import reward,GAMMA,BETA
def main():
 import h5py
 from storage import push_progress
 root=WORK/'offline';info=json.loads((root/'complete.json').read_text());reports=json.loads((root/'episodes.json').read_text())
 assert info['accepted_demonstrations']==len(reports)==50
 assert sha(root/'replay.npz')==info['replay_sha256']
 with np.load(root/'replay.npz',allow_pickle=False) as archive:
  data={k:archive[k] for k in ('actions','base_actions','next_base_actions','rewards','dones','mc_returns','images','next_images')}
 offset=0;reward_error=return_error=telescoping_error=0.
 with h5py.File(source('D1'),'r') as native:
  for report in reports:
   n=report['transitions'];end=offset+n;g=native['data'][report['demo']];phi=report['phi'];assert len(phi)==n+1
   assert min(phi)>=0 and max(phi)<=4
   np.testing.assert_array_equal(data['actions'][offset:end],g['actions'][1:n+1].astype(np.float32))
   images=np.stack([g['obs'][k][:n+1][:,::-1,::-1,:] for k in ('agentview_rgb','eye_in_hand_rgb')],axis=1)
   np.testing.assert_array_equal(data['images'][offset:end],images[:-1])
   np.testing.assert_array_equal(data['next_images'][offset:end],images[1:])
   np.testing.assert_array_equal(data['next_base_actions'][offset:end-1],data['base_actions'][offset+1:end])
   np.testing.assert_array_equal(data['next_base_actions'][end-1],np.zeros(7,np.float32))
   np.testing.assert_array_equal(data['dones'][offset:end],np.r_[np.zeros(n-1),1])
   expected=np.asarray([reward(i==n-1,phi[i],phi[i+1],i==n-1) for i in range(n)])
   reward_error=max(reward_error,float(np.max(np.abs(data['rewards'][offset:end]-expected))))
   running=0.
   for i in range(n-1,-1,-1):
    running=expected[i]+GAMMA*running
    return_error=max(return_error,abs(float(data['mc_returns'][offset+i])-running))
   telescoping_error=max(telescoping_error,abs(float(data['mc_returns'][offset])-(GAMMA**(n-1)-BETA*phi[0])))
   offset=end
 assert offset==info['transitions']==5055
 assert max(reward_error,return_error,telescoping_error)<1e-6
 proof=dict(passed=True,demonstrations=50,transitions=offset,original_actions_and_both_camera_frames_exact=True,
  nonterminal_next_base_cache_exact=True,terminal_next_base_zero=True,maximum_reward_error=reward_error,
  maximum_mc_return_error=return_error,maximum_telescoping_identity_error=telescoping_error,
  replay_sha256=info['replay_sha256'],base_weights_sha256=info['base_weights_sha256'],final_seeds_used=False)
 write(WORK/'evidence/native_replay_reward_return_audit.json',proof);print(proof,flush=True)
 push_progress('exp: independently audit all 5055 native transitions, camera frames and shaped returns')
if __name__=='__main__':main()
