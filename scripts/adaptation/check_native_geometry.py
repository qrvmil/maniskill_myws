"""CPU-only check of original native D1 state geometry and LIBERO success labels."""
from types import SimpleNamespace
from common import *
def main():
 import h5py
 from libero.libero import get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero.utils.utils import postprocess_model_xml
 from maniskill_myws.pld.libero_demonstrations import relocate_legacy_assets
 from maniskill_myws.pld.libero_progress import task_progress
 from shaping import potential
 spec=TASKS['D1'];bddl=Path(get_libero_path('bddl_files'))/spec['suite']/(spec['name']+'.bddl')
 control=ControlEnv(str(bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,camera_names=[])
 low=control.env;outer=SimpleNamespace(env=control);reports=[]
 with h5py.File(source('D1'),'r') as f:
  for name in sorted(f['data'],key=lambda n:int(n.split('_')[-1])):
   g=f['data'][name];states=g['states'][:];actions=g['actions'][:]
   control.reset();xml=postprocess_model_xml(str(g.attrs['model_file']),{})
   xml,assets=relocate_legacy_assets(xml,get_libero_path('assets'));low.reset_from_xml_string(xml)
   sim=low.sim;progress=[];success=[]
   for i in range(len(actions)):
    sim.set_state_from_flattened(states[min(i+1,len(states)-1)]);sim.forward()
    if i==len(states)-1:low.step(np.asarray(actions[-1],np.float32).tolist())
    p=task_progress(outer,{'robot0_eef_pos':g['obs']['ee_pos'][i]});progress.append(p)
    success.append(bool(low._check_success()))
   first=next((i for i,x in enumerate(success) if x),None)
   phi=[potential(p,progress[0]['bowl_height']) for p in progress]
   assert np.isfinite(phi).all() and min(phi)>=0 and max(phi)<=4
   accepted=bool(float(g['rewards'][-1])>0 and first is not None and first>0)
   reports.append(dict(demo=name,accepted=accepted,native_terminal_reward=float(g['rewards'][-1]),first_saved_state_success=first,potentials=phi))
   print('NATIVE_GEOMETRY',name,accepted,first,flush=True)
 control.close()
 write(WORK/'evidence/D1_native_geometry_audit.json',dict(source_sha256=sha(source('D1')),demonstrations=50,
   accepted=sum(r['accepted'] for r in reports),renderer=False,policy_actions_not_computed=True,episodes=reports))
 assert any(r['accepted'] for r in reports),'No native successful D1 demos verified'
 print('NATIVE_D1_GEOMETRY_AND_SUCCESS_PASS',sum(r['accepted'] for r in reports),flush=True)
if __name__=='__main__':main()
