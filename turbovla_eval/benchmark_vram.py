"""Fresh-process GPU sampling matching paper Appendix B.2; no simulator in child."""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from eval_utils import ROOT, gpu_used_mib, memory_summary, write_json, validate_idle_memory


def worker(precision, destination):
    import numpy as np
    from eval_utils import load_policy
    policy=load_policy(precision)
    import torch
    observation=dict(np.load(ROOT/'assets/sample_observation.npz'))
    observation_meta=json.loads((ROOT/'assets/sample_observation.json').read_text())
    instruction=observation_meta['instruction']
    from turbovla.evaluation.policy import rotate_libero_image,state_from_libero_obs
    samples,states=policy._build_batch([rotate_libero_image(observation['agentview_image'])],
        [rotate_libero_image(observation['robot0_eye_in_hand_image'])],[state_from_libero_obs(observation)])
    samples,states=policy._prepare_model_inputs(samples,states)
    model=policy.model
    with torch.inference_mode():
        for _ in range(10):
            output=model([instruction],samples,states)
            torch.cuda.synchronize()
            del output
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for _ in range(50):
            torch.cuda.synchronize()
            start=time.perf_counter()
            output=model([instruction],samples,states)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            assert output.shape==(1,12,7) and torch.isfinite(output).all()
            del output
    result=dict(precision=precision,parameters=sum(p.numel() for p in model.parameters()),
        parameter_gib=sum(p.numel()*p.element_size() for p in model.parameters())/1024**3,
        torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        median_ms=float(np.median(timings)),p90_ms=float(np.percentile(timings,90)),mean_ms=float(np.mean(timings)),
        timings_ms=timings,warmup=10,timed_calls=50,batch_size=1,action_horizon=12,
        input='Real settled D0 observation saved before rollout; GPU-resident normalized dual-view input',
        includes='BERT + DINOv3 + fusion + action head; tokenization included by official forward',
        excludes='Simulator, image preparation/transfer and CPU action denormalization; output remains on GPU',
        observation_metadata=observation_meta,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda)
    write_json(destination,result)


def benchmark(precision='bf16',output=None):
    output=Path(output or ROOT/'results'/f'vram_{precision}')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    if not (ROOT/'assets/sample_observation.npz').is_file() or not (ROOT/'assets/sample_observation.json').is_file():
        subprocess.run([sys.executable,str(ROOT/'prepare_observation.py')],check=True)
    # Parent has no torch import/context. Start sampler before policy initialization.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if any(line.strip().isdigit() for line in active.splitlines()):
        raise RuntimeError('GPU compute process already active. Shut down GPU Jupyter kernels/evaluation before measuring VRAM.')
    idle=gpu_used_mib()
    validate_idle_memory(idle)
    samples=[]
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=50'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    started=time.time()
    def collect():
        with open(output/'memory_samples.jsonl','w') as stream:
            for line in sampler.stdout:
                try: value=float(line.strip().splitlines()[0])
                except ValueError: continue
                sample=dict(time_s=time.time()-started,used_mib=value)
                samples.append(sample)
                stream.write(json.dumps(sample)+'\n');stream.flush()
    thread=threading.Thread(target=collect,daemon=True);thread.start()
    try:
        with open(output/'worker.log','w') as log:
            child=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--precision',precision,
                '--output',str(output/'worker.json')],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:
            raise RuntimeError(f'Benchmark failed: see {output / "worker.log"}')
    finally:
        sampler.terminate();sampler.wait(timeout=10);thread.join(timeout=10)
    result=json.loads((output/'worker.json').read_text())
    result.update(memory_summary(samples,idle))
    result.update(sample_interval_ms=50,sample_count=len(samples),
        measurement='Peak whole-device nvidia-smi memory minus idle, including loading, warmup and timed inference')
    write_json(output/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker: worker(a.precision,a.output)
    else: print(json.dumps(benchmark(a.precision,a.output),indent=2))
