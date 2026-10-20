"""Confirm the selected early teacher under canonical fresh/restored sensor resets."""
from common import *
def main():
 import torch
 from evaluate import validation_teacher
 from storage import publish,push_progress
 teacher=json.loads((WORK/'teacher.json').read_text());checkpoint=Path(teacher['checkpoint'])
 weights=sha(checkpoint/'model.safetensors');folder=WORK/'teacher_recheck';path=folder/'validation.json'
 if path.exists():
  result=json.loads(path.read_text());assert result['weights_sha256']==weights
 else:
  torch.set_num_threads(4)
  model=load_model(checkpoint).to('cuda');result=validation_teacher(model,checkpoint)
  result.update(weights_sha256=weights,reset_protocol='fresh and saved states both use set_init_state before settling; sensor cache refreshed identically',original_selection_successes=teacher['validation']['successes'],selected_step_unchanged=teacher['step'],final_seeds_used=False)
  write(path,result)
 assert result['successes']>0,'Selected early teacher needs nonzero validation success under canonical resets'
 teacher['canonical_validation']=result;write(WORK/'teacher.json',teacher)
 write(WORK/'evidence/teacher_canonical_validation.json',result)
 publish(folder,'validation/teacher_canonical_reset',full_readback=True)
 push_progress('exp: confirm unchanged early teacher on canonical-reset validation scenes')
if __name__=='__main__':main()
