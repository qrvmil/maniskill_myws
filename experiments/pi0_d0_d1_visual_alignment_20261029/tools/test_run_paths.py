"""Prevent new runs from writing historical results or using old completion markers."""
from pathlib import Path
import importlib.util,sys
import pytest
ROOT=Path(__file__).resolve().parents[3]

def load(monkeypatch,**values):
 for name in ['PI0_WORK','PI0_HF_REPO','PI0_EXPERIMENT_DIR','PI0_MICROBATCH','PI0_VISUAL_ALIGNMENT']:monkeypatch.delenv(name,raising=False)
 for name,value in values.items():monkeypatch.setenv(name,value)
 spec=importlib.util.spec_from_file_location('isolated_adaptation_common',ROOT/'scripts/adaptation/common.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_new_destinations_and_batch_preserve_original_budgets(monkeypatch,tmp_path):
 m=load(monkeypatch,PI0_WORK=str(tmp_path/'run'),PI0_HF_REPO='qrvmil/new-experiment',PI0_EXPERIMENT_DIR='new_experiment',PI0_MICROBATCH='1',PI0_VISUAL_ALIGNMENT='1')
 assert m.WORK==tmp_path/'run' and m.HF_REPO=='qrvmil/new-experiment'
 assert m.EXPERIMENT==ROOT/'experiments/new_experiment'
 assert m.MICROBATCH==1 and m.BATCH==8
 assert [m.SFT[x]['updates'] for x in ['d0','full','expert']]==[3000,1500,1500]

def test_legacy_defaults_are_preserved(monkeypatch):
 m=load(monkeypatch)
 assert m.WORK==Path('/workspace/pi0_three')
 assert m.HF_REPO=='qrvmil/pi0-d0-d1-three-adaptations-20261020'
 assert m.EXPERIMENT==ROOT/'experiments/pi0_d0_d1_three_adaptations'
 assert m.MICROBATCH==2 and m.BATCH==8

def test_alignment_cannot_silently_write_to_legacy_experiment(monkeypatch):
 with pytest.raises(ValueError,match='isolated'):load(monkeypatch,PI0_VISUAL_ALIGNMENT='1')

def test_unsupported_microbatch_rejected(monkeypatch):
 with pytest.raises(ValueError,match='microbatch'):load(monkeypatch,PI0_MICROBATCH='3')
