#!/usr/bin/env python3
"""Backfill selected videos after the serial scientific GPU stages finish."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/workspace/audit-run')


def main():
    while True:
        status=subprocess.run(['supervisorctl','status','pi0_audit_pipeline'],capture_output=True,text=True).stdout.split()
        if len(status)<2:raise RuntimeError('Cannot verify audit pipeline status')
        if status[1]=='EXITED':break
        if status[1]!='RUNNING':raise RuntimeError('Audit pipeline is not running: '+status[1])
        time.sleep(5)
    for model in ('0','positive_control','lora_initialization','500','1000','2000','3001'):
        if not (WORK/f'eval/{model}/complete.json').exists():
            raise RuntimeError(f'Audit pipeline exited before completing {model}')
    for model in ('0','positive_control'):
        completed=WORK/f'eval/{model}/video_replay/complete.json'
        if completed.exists():continue
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/pld/audit_pi0.py'),'eval',
                        '--model',model,'--video-replay'],cwd=ROOT,check=True)
        if not completed.exists():raise RuntimeError('Replay exited without complete evidence')
    (WORK/'video_replays_complete.json').write_text(json.dumps({'models':['0','positive_control'],'status':'COMPLETED'},indent=2)+'\n')
    print('ALL_REQUIRED_VIDEO_REPLAYS_COMPLETE',flush=True)


if __name__=='__main__':main()
