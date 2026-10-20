"""A completed supervised one-shot job is a valid pipeline dependency."""
import subprocess,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/adaptation'))
import pipeline

def test_completed_service_status_has_nonzero_cli_exit(monkeypatch):
 monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,3,'pi0_three_d0 EXITED Oct 20 05:02 PM\n',''))
 assert pipeline.service_status('pi0_three_d0')=='EXITED'

def test_unavailable_supervisor_is_not_treated_as_completion(monkeypatch):
 monkeypatch.setattr(pipeline.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1,'','connection refused'))
 with pytest.raises(RuntimeError,match='connection refused'):pipeline.service_status('pi0_three_d0')
