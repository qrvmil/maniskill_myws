"""Prove actor-only evaluation equals the diagnostic deterministic-actor wrapper."""
from common import *
def main():
 import torch
 from evaluate import FlowModel,PairedVideoEnv
 from maniskill_myws.pld.libero_backend import ChunkedBasePolicy,convert_observation
 from maniskill_myws.pld.libero_runner import run_episode,residual_policy,actor_residual
 from maniskill_myws.pld.sac import ResidualSAC
 from storage import push_progress
 torch.set_num_threads(4);torch.manual_seed(0)
 source=WORK/'rl/after_teacher_warmup/residual.pt'
 model=load_model(WORK/'sft/d0/3000').to('cuda');agent=ResidualSAC.load(source,device='cuda')
 flow=FlowModel(model,'D1');base=ChunkedBasePolicy(flow,replan_steps=5)
 env=PairedVideoEnv('D1','actor_adapter_check',video=False)
 old=residual_policy(agent,{'otf_rollout_actions':1},'deterministic_actor',seed=9023004);direct=actor_residual(agent)
 before={n:tensor_hash(p) for n,p in agent.actor.named_parameters()}
 try:
  raw,_=env.reset(seed=23004);obs=convert_observation(raw,env.prompt);rng=np.random.default_rng(23004)
  for _ in range(10):
   action=rng.uniform(-1,1,7).astype(np.float32);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
   a=old(obs,action);b=direct(obs,action)
   np.testing.assert_array_equal(a,b)
   assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
  a,ta=run_episode(env,base,seed=23004,residual=old,residual_scale=2.)
  noises=list(flow.noise_hashes)
  b,tb=run_episode(env,base,seed=23004,residual=direct,residual_scale=2.)
  for key in ('reset_hash','trajectory_hash','image_hash','length','success'):assert a[key]==b[key],key
  assert noises==flow.noise_hashes
  for x,y in zip(ta,tb):np.testing.assert_array_equal(x['action'],y['action'])
 finally:env.close()
 assert before=={n:tensor_hash(p) for n,p in agent.actor.named_parameters()}
 proof=dict(passed=True,residual_checkpoint_sha256=sha(source),task='D1',seed=23004,random_base_action_cases=10,
  controller_actions_bitwise_equal=True,cpu_cuda_rng_unchanged=True,actor_parameters_unchanged=True,
  reset_hash=a['reset_hash'],trajectory_hash=a['trajectory_hash'],image_hash=a['image_hash'],length=a['length'],success=a['success'],
  diagnostic_wrapper_wall_seconds=a['wall_seconds'],actor_only_wall_seconds=b['wall_seconds'],final_seeds_used=False)
 write(WORK/'evidence/actor_only_evaluation_equivalence.json',proof);print(proof,flush=True)
 push_progress('exp: prove deterministic actor-only evaluation preserves actions trajectories images and RNG')
if __name__=='__main__':main()
