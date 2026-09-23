#!/usr/bin/env python3
"""Serial GPU pipeline; never changes settings in response to evaluation outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import time
from maniskill_myws.pld.multitask_data import WORK,ROOT,train_config


def run(cmd):
    print('START',cmd,flush=True)
    subprocess.run([sys.executable,'-u',*cmd],cwd=ROOT,check=True)


def main():
    while not (WORK/'C/binding.json').exists():
        status=subprocess.check_output(['supervisorctl','status','multitask_prepare'],text=True).split()[1]
        if status not in ('RUNNING','STARTING'):raise RuntimeError('Preparation stopped before C completed')
        time.sleep(10)
    # Reuse validation must pass against recorded source/statistics provenance.
    for variant in ('A','B','C'):
        run(['scripts/prepare_multitask_sft.py','prepare','--variant',variant])
    run(['scripts/check_multitask_data.py'])
    for variant in ('A','B','C'):
        if not (WORK/variant/'training_complete.json').exists():
            run(['scripts/train_multitask_sft.py','--variant',variant])
    # All primary runs finish before held-out evaluation; no adaptive selection.
    for variant in ('A','B','C'):
        checkpoint=train_config(variant).checkpoint_dir/'3001'
        output=WORK/'eval'/variant/'3001'
        if not (output/'image_sensitivity.json').exists():
            run(['scripts/eval_multitask_sft.py','--variant',variant,'--checkpoint',str(checkpoint),
                 '--task','all','--output',str(output),'--image-sensitivity'])
    for updates in (0,500,1000,2000):
        for variant in ('A','B','C'):
            checkpoint=train_config(variant).checkpoint_dir/str(updates)
            output=WORK/'eval'/variant/str(updates)
            if not (output/'D2/complete.json').exists():
                run(['scripts/eval_multitask_sft.py','--variant',variant,'--checkpoint',str(checkpoint),
                     '--task','D0,D1,D2','--videos','0','--output',str(output)])
    (WORK/'gpu_pipeline_complete.json').write_text(json.dumps({'primary':True,'trajectory':True})+'\n')

if __name__=='__main__':main()
