"""Native successful D1 replay, cached new-D0 actions and state-based shaping.

Native camera/proprioception and actions are retained verbatim. Saved pre-action
states are restored only to compute geometry/contact potentials and goal labels;
this is not a new demonstration rollout. The last post-action state is obtained
by executing the recorded final action from its saved pre-action state.
"""
from common import *
from shaping import potential,reward,returns

def main():
 import h5py,torch
 from libero.libero import get_libero_path
 from libero.libero.utils.utils import postprocess_model_xml
 from maniskill_myws.pld.libero_demonstrations import native_observations,relocate_legacy_assets
 from maniskill_myws.pld.libero_backend import LiberoEnv,ChunkedBasePolicy,ActionContract
 from maniskill_myws.pld.libero_progress import task_progress
 from maniskill_myws.pld.replay_buffer import ReplayBuffer
 from evaluate import FlowModel
 from storage import publish,push_progress
 torch.set_num_threads(4)
 root=WORK/'offline';root.mkdir(exist_ok=True)
 base_path=WORK/'sft/d0/3000';base_hash=sha(base_path/'model.safetensors')
 if (root/'complete.json').exists():
  old=json.loads((root/'complete.json').read_text());assert old['base_weights_sha256']==base_hash;assert old['replay_sha256']==sha(root/'replay.npz');return
 model=load_model(base_path).to('cuda');flow=FlowModel(model,'D1');base=ChunkedBasePolicy(flow,replan_steps=5)
 outer=LiberoEnv(TASKS['D1'],render_size=128);low=outer.env.env
 buffer=ReplayBuffer(20000,8,7,image_shape=(2,128,128,3));reports=[]
 with h5py.File(source('D1'),'r') as f:
  for demo_index,name in enumerate(sorted(f['data'],key=lambda s:int(s.split('_')[-1]))):
   g=f['data'][name];observations=list(native_observations(g,outer.prompt));states=g['states'][:];actions=g['actions'][:]
   low.reset();xml=postprocess_model_xml(str(g.attrs['model_file']),{})
   xml,assets=relocate_legacy_assets(xml,get_libero_path('assets'));low.reset_from_xml_string(xml)
   sim=low.sim;progress=[];success=[]
   # obs[i] is post action[i], so the corresponding saved state is states[i+1].
   for i,obs in enumerate(observations):
    sim.set_state_from_flattened(states[min(i+1,len(states)-1)]);sim.forward()
    if i==len(states)-1:
     low.step(np.asarray(actions[-1],np.float32).tolist())
    progress.append(task_progress(outer,obs['raw']));success.append(bool(low._check_success()))
   first=next((i for i,v in enumerate(success) if v),None)
   accepted=bool(float(g['rewards'][-1])>0 and first is not None and first>0)
   report=dict(demo=name,accepted=accepted,native_terminal_reward=float(g['rewards'][-1]),
    first_saved_state_success=first,source_sha256=sha(source('D1')),
    initial_xml_sha256=hashlib.sha256(str(g.attrs['model_file']).encode()).hexdigest(),relocated_asset_count=len(assets))
   if accepted:
    selected=observations[:first+1];base.reset(1000000+demo_index);bases=[];inference=[]
    original=flow.infer
    def logged(raw):
     result=original(raw);inference.append(dict(observation_index=len(bases),noise_sha256=flow.noise_hashes[-1],first5_preclip=np.asarray(result['actions'][:5]).tolist()));return result
    flow.infer=logged
    for obs in selected:bases.append(base.act(obs['raw']))
    flow.infer=original
    phi=[potential(p,progress[0]['bowl_height']) for p in progress[:first+1]];transitions=[]
    for i in range(first):
     o,nxt=selected[i:i+2];done=i==first-1;a=np.asarray(actions[i+1],np.float32)
     unit=(a-bases[i])/2
     assert np.max(np.abs(unit))<=1
     np.testing.assert_allclose(ActionContract(2).compose(bases[i],unit),a,atol=2e-7)
     transitions.append(dict(obs=o['state'],action=a,base_action=bases[i],reward=reward(done,phi[i],phi[i+1],done),
      next_obs=nxt['state'],next_base_action=bases[i+1],done=done,images=o['images'],next_images=nxt['images']))
    mc=returns(transitions)
    for t,ret in zip(transitions,mc):buffer.add(**t,mc_return=ret)
    report.update(transitions=len(transitions),phi=phi,inference_calls=inference,base_actions_sha256=hashlib.sha256(np.asarray(bases).tobytes()).hexdigest(),
     max_absolute_correction=float(np.max(np.abs(np.asarray(actions[1:first+1])-np.asarray(bases[:-1])))),reachable_fraction=1.)
   reports.append(report);write(root/'episodes.json',reports)
   print('NATIVE_OFFLINE_REPLAY',name,accepted,first,len(buffer),flush=True)
 outer.close()
 if not len(buffer):raise RuntimeError('No native D1 demo passes saved-state LIBERO success verification')
 buffer.save(root/'replay.npz',source='native D1',base_weights_sha256=base_hash,normalization_sha256=sha(NORM),gamma=.99,beta=.1)
 write(root/'complete.json',dict(source_revision='f13aa24a3da8c43c7225569f28c562979fa0e35a',source_sha256=sha(source('D1')),
  base_weights_sha256=base_hash,normalization_sha256=sha(NORM),accepted_demonstrations=sum(r['accepted'] for r in reports),
  original_demonstrations=50,transitions=len(buffer),replay_sha256=sha(root/'replay.npz'),
  observation_source='original native HDF5; no generated observations',potential_source='restored matching original XML saved states; final recorded action for final post-action state',
  action_alignment='native obs[i] to actions[i+1]',residual_scale=2,teacher_not_used=True))
 publish(root,'offline',full_readback=True);push_progress('exp: preserve native D1 replay recomputed for the new frozen D0 base')
if __name__=='__main__':main()
