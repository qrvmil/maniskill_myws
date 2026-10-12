"""Device memory including EGL rendering, sampled outside the evaluation process."""
import json
import subprocess
import sys
import threading
import time
from eval_utils import ROOT,gpu_used_mib,memory_summary,write_json

if __name__=='__main__':
    idle=gpu_used_mib();samples=[];started=time.time()
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=100'],stdout=subprocess.PIPE,text=True)
    def collect():
        with open(ROOT/'results/rollout_memory_samples.jsonl','w') as f:
            for line in sampler.stdout:
                try: m=float(line.strip())
                except ValueError: continue
                row=dict(time_s=time.time()-started,used_mib=m);samples.append(row)
                f.write(json.dumps(row)+'\n');f.flush()
    ROOT.joinpath('results').mkdir(exist_ok=True)
    thread=threading.Thread(target=collect,daemon=True);thread.start()
    try:
        child=subprocess.run([sys.executable,str(ROOT/'run_eval.py'),*sys.argv[1:]])
    finally:
        sampler.terminate();sampler.wait();thread.join(timeout=10)
        result=memory_summary(samples,idle)
        result.update(command_arguments=sys.argv[1:],duration_s=time.time()-started,sample_interval_ms=100,includes='Full rollout process: policy + EGL LIBERO simulator + video recording',child_returncode=child.returncode)
        write_json(ROOT/'results/rollout_memory.json',result)
    sys.exit(child.returncode)
