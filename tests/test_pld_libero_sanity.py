"""Audit-only controls must not weaken historical alignment requirements."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest


def audit_config():
    return json.loads(Path('configs/pld_libero/pi0_sanity_audit.json').read_text())


def test_fixed_seeds_across_models_and_update_counters():
    from maniskill_myws.pld.libero_sanity import evaluation_seeds, completed_updates
    cfg=audit_config()
    for step in cfg['optimizer_updates']:
        assert evaluation_seeds(cfg,step)==tuple(range(9000,9050))
    assert evaluation_seeds(cfg,'positive_control')==tuple(range(9000,9050))
    assert completed_updates(499,500)==500
    assert completed_updates(3000,3001)==3001
    with pytest.raises(ValueError):completed_updates(500,500)
    bad=copy.deepcopy(cfg);bad['seeds'][1]=9000
    with pytest.raises(ValueError):evaluation_seeds(bad,0)
    with pytest.raises(ValueError):evaluation_seeds(cfg,501)


def test_source_only_training_and_normalization_binding(tmp_path):
    from maniskill_myws.pld.libero_sanity import require_d0_binding
    from maniskill_myws.pld.libero_protocol import file_sha256, task_key
    norm=tmp_path/'norm_stats.json';norm.write_text('{}')
    cfg=audit_config();d0=task_key(cfg['tasks'][0]);d1=task_key(cfg['tasks'][1])
    binding={'training_tasks':[d0],'num_demonstrations':50,'transitions':5832,
             'normalization_path':str(norm),'normalization_sha256':file_sha256(norm),
             'pretrained_checkpoint':'gs://openpi-assets/checkpoints/pi0_base'}
    require_d0_binding(binding,cfg)
    for change in ({'training_tasks':[d0,d1]}, {'training_tasks':[d1]},
                   {'num_demonstrations':49}, {'pretrained_checkpoint':'gs://openpi-assets/checkpoints/pi05_libero'}):
        with pytest.raises(ValueError):require_d0_binding(dict(binding,**change),cfg)
    norm.write_text('{"changed":true}')
    with pytest.raises(ValueError):require_d0_binding(binding,cfg)


def test_positive_control_cannot_use_d0_statistics():
    from maniskill_myws.pld.libero_sanity import model_contract
    cfg=audit_config()
    own={'normalization_path':'d0/norm_stats.json'}
    for step in cfg['optimizer_updates']:
        assert model_contract(cfg,step,own)['normalization_path']==own['normalization_path']
    official=model_contract(cfg,'positive_control',own)
    assert official['config']=='pi05_libero'
    assert official['normalization_path'] is None  # checkpoint-provided only
    assert official['checkpoint']==cfg['positive_control']['checkpoint']
    assert official['purpose']=='positive_control_only'


def test_real_inference_adapter_checks_prompt_at_tokenizer():
    pytest.importorskip('openpi')
    from maniskill_myws.pld.libero_sanity import PromptCheckedPolicy
    from maniskill_myws.pld.libero_experiment import AlignedOpenPIModel
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.models.model import ModelType
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from types import SimpleNamespace
    tokenizer=transforms.TokenizePrompt(PaligemmaTokenizer(48))
    # Real adapter and official preprocessing/tokenization; expensive neural network is replaced.
    class Policy:
        _input_transform=transforms.compose([LiberoInputs(ModelType.PI0),tokenizer])
        def infer(self,obs,noise):
            self.last=self._input_transform(obs)
            return {'actions':np.zeros((50,7),np.float32)}
    policy=Policy()
    expected=['']
    checked=PromptCheckedPolicy(policy,lambda:expected[0])
    model=AlignedOpenPIModel.__new__(AlignedOpenPIModel)
    model.policy=checked;model.noise_shape=(50,32);model.inference_seconds=[]
    model.run=SimpleNamespace(begin_cuda_phase=lambda:None,end_cuda_phase=lambda _:None)
    im=np.zeros((16,16,3),np.uint8)
    raw=dict(agentview_image=im,robot0_eye_in_hand_image=im,
             robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0,0,0,1]),robot0_gripper_qpos=np.zeros(2))
    tokens=[]
    for task in audit_config()['tasks']:
        expected[0]=model.prompt=task['name'].replace('_',' ')
        model.reset(9000);model.infer(raw)
        tokens.append(policy.last['tokenized_prompt'].copy())
    assert not np.array_equal(*tokens)
    assert checked.calls==2
    model.prompt=audit_config()['tasks'][0]['name'].replace('_',' ')
    with pytest.raises(ValueError,match='prompt'):model.infer(raw)


def test_primary_step_zero_is_official_weights_and_auxiliary_is_lora_initialization():
    from maniskill_myws.pld.libero_sanity import model_contract
    cfg=audit_config();binding={'normalization_path':'d0/norm_stats.json'}
    zero=model_contract(cfg,0,binding)
    init=model_contract(cfg,'lora_initialization',binding)
    assert zero['weight_source']=='official_pretrained'
    assert zero['use_lora'] is False
    assert init['weight_source']=='saved_training_state'
    assert init['checkpoint_updates']==0
    assert init['use_lora'] is True
    for step in (500,1000,2000,3001):
        later=model_contract(cfg,step,binding)
        assert later['checkpoint_updates']==step
        assert later['use_lora'] is True
    assert evaluation_seeds_for_auxiliary(cfg)==tuple(range(9000,9050))


def evaluation_seeds_for_auxiliary(cfg):
    from maniskill_myws.pld.libero_sanity import evaluation_seeds
    return evaluation_seeds(cfg,'lora_initialization')


def test_checkpoint_callback_saves_actual_completed_updates_only():
    from maniskill_myws.pld.libero_sanity import save_selected_update
    from types import SimpleNamespace
    written=[]
    def native_save(manager,state,loader,directory):
        written.append((directory,state.step))
    for loop_index in (0,498,499,500,999,1999,3000):
        state=SimpleNamespace(step=loop_index+1)
        save_selected_update(native_save,None,state,None,loop_index)
    assert written==[(500,500),(1000,1000),(2000,2000),(3001,3001)]
    with pytest.raises(ValueError):save_selected_update(native_save,None,SimpleNamespace(step=499),None,499)


def test_summary_rejects_missing_duplicate_or_mismatched_paired_scenes():
    from maniskill_myws.pld.libero_sanity import summarize_rows, paired_change
    rows=[dict(seed=s,reset_hash=str(s),success=i<10,length=100+i) for i,s in enumerate(range(9000,9050))]
    result=summarize_rows(rows)
    assert result['successes']==10 and result['success_rate']==.2
    assert result['mean_episode_length']==124.5
    assert result['wilson_95ci'][0]<.2<result['wilson_95ci'][1]
    with pytest.raises(ValueError):summarize_rows(rows[:-1])
    bad=copy.deepcopy(rows);bad[1]['seed']=9000
    with pytest.raises(ValueError):summarize_rows(bad)
    later=copy.deepcopy(rows);later[0]['success']=False;later[10]['success']=True
    change=paired_change(rows,later)
    assert change['gained']==change['lost']==1 and change['delta_pp']==0
    later[0]['reset_hash']='wrong'
    with pytest.raises(ValueError):paired_change(rows,later)


def test_paired_perfect_gain_has_uncertainty_but_identity_has_none():
    from maniskill_myws.pld.libero_sanity import paired_change
    zero=[dict(seed=s,reset_hash=str(s),success=False,length=220) for s in range(9000,9050)]
    one=[dict(r,success=True,length=100) for r in zero]
    gain=paired_change(zero,one)
    assert gain['delta_pp']==100 and gain['interval_95_pp'][0]<100
    assert gain['interval_95_pp'][1]==100
    loss=paired_change(one,zero)
    assert loss['delta_pp']==-100 and loss['interval_95_pp'][1]>-100
    assert paired_change(zero,zero)['interval_95_pp']==[0.,0.]
    assert paired_change(zero,copy.deepcopy(zero))['interval_95_pp'][1]>0
