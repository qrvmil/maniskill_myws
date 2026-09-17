#!/usr/bin/env python3
"""Serial GPU execution; full positive control must complete before D0 training."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/workspace/audit-run')


def main():
    control=WORK/'eval/positive_control/complete.json'
    while not control.exists():
        metadata=WORK/'runtime/eval_positive_control/metadata.json'
        if metadata.exists() and json.loads(metadata.read_text())['status']=='FAILED':
            raise RuntimeError('Positive control failed; inspect its error evidence before continuing')
        time.sleep(5)
    # The completion file precedes Python/CUDA teardown; await the external
    # supervisor process exit before the next GPU model is constructed.
    while True:
        status=subprocess.run(['supervisorctl','status','pi0_audit'],capture_output=True,text=True).stdout.split()
        if len(status)<2:raise RuntimeError('Cannot verify positive-control process exit')
        if status[1]=='EXITED':break
        if status[1]!='RUNNING':raise RuntimeError('Unexpected positive-control process state: '+status[1])
        time.sleep(1)
    result=json.loads((control.parent/'D1_summary.json').read_text())
    if result['n']!=50 or result['successes']<=2:
        raise RuntimeError('Official D1 positive control near zero: investigate bridge before specialization claims')
    print('POSITIVE_CONTROL_PASSED',result,flush=True)
    stages=[('eval','0'),('train',None),('eval','lora_initialization'),
            ('eval','500'),('eval','1000'),('eval','2000'),('eval','3001')]
    for mode,model in stages:
        completed=WORK/('training_complete.json' if mode=='train' else f'eval/{model}/complete.json')
        if completed.exists():
            print('ALREADY_COMPLETED',mode,model,flush=True)
            continue
        cmd=[sys.executable,'-u',str(ROOT/'scripts/pld/audit_pi0.py'),mode]
        if model is not None:cmd.extend(['--model',model])
        print('STARTING',mode,model,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
        if not completed.exists():raise RuntimeError('Stage exited without complete evidence')
    print('ALL_AUDIT_GPU_STAGES_COMPLETE',flush=True)

if __name__=='__main__':main()
