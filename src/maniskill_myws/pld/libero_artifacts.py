"""Immutable run directories and actual machine/runtime measurements."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import importlib.metadata
import os
import resource
import shutil
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import traceback


def memory_fraction(budget_gib,total_bytes):
    import math
    if not math.isfinite(budget_gib) or budget_gib<=0 or total_bytes<=0:
        raise ValueError('GPU memory budget and capacity must be positive')
    return min(1.0,float(budget_gib)*1024**3/total_bytes)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def command_output(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else p.stderr.strip()


class RunArtifacts(AbstractContextManager):
    def __init__(self, output, config):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        for sub in ['logs','checkpoints','eval']:
            (self.path/sub).mkdir()
        write_json(self.path/'config.json', config)
        root=Path(__file__).resolve().parents[3]
        snapshot=self.path/'code_snapshot'
        for directory,pattern in [('src/maniskill_myws/pld','*.py'),('scripts/pld','*libero*.py')]:
            for src in (root/directory).glob(pattern):
                dst=snapshot/src.relative_to(root)
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        write_json(self.path/'package_versions.json',dict(sorted(
            (d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])))
        (self.path/'git_diff.patch').write_text(command_output(['git','diff','HEAD','--',
            'src/maniskill_myws/pld','scripts/pld','configs/pld_libero']))
        import torch
        self.torch = torch
        self.cuda_peaks = {}
        self.start = time.perf_counter()
        self.stop_event = threading.Event()
        self.samples = []
        self.meta = dict(date_utc=datetime.now(timezone.utc).isoformat(),
            command=shlex.join([sys.executable,*sys.argv]), config=config,
            git_commit=command_output(['git','rev-parse','HEAD']),
            git_status=command_output(['git','status','--short']),
            gpu=command_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv']),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            environment={k:os.environ[k] for k in ['JAX_PLATFORMS','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE','MUJOCO_GL','HF_LEROBOT_HOME','PYTHONPATH','TORCH_COMPILE_DISABLE','OMP_NUM_THREADS'] if k in os.environ},
            status='RUNNING', checkpoint=None)
        write_json(self.path/'metadata.json',self.meta)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.thread = threading.Thread(target=self._monitor,daemon=True)
        self.thread.start()

    def capture_cuda_peak(self):
        cuda=self.torch.cuda
        peaks={'allocated_bytes':cuda.max_memory_allocated() if cuda.is_available() else 0,
               'reserved_bytes':cuda.max_memory_reserved() if cuda.is_available() else 0}
        for key,value in peaks.items():
            self.cuda_peaks[key]=max(value,self.cuda_peaks.get(key,0))
        return peaks

    def begin_cuda_phase(self):
        cuda=self.torch.cuda
        if cuda.is_available():cuda.synchronize()
        self.capture_cuda_peak()
        if cuda.is_available():
            cuda.reset_peak_memory_stats()
            return cuda.memory_allocated()
        return 0

    def end_cuda_phase(self,name):
        peaks=self.capture_cuda_peak()
        saved=self.meta.setdefault('cuda_phase_peaks',{}).setdefault(name,{})
        for key,value in peaks.items():saved[key]=max(value,saved.get(key,0))
        return peaks

    def _monitor(self):
        while not self.stop_event.is_set():
            out=command_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'])
            try:
                sample=dict(seconds=time.perf_counter()-self.start,used_mib=int(out.splitlines()[0]))
                self.samples.append(sample)
                # Keep raw samples even if supervisor terminates the process.
                with (self.path/'memory_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample)+'\n')
            except ValueError:
                pass
            self.stop_event.wait(.5)

    def __exit__(self, typ, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.meta.update(status='FAILED' if exc else 'COMPLETED',
                         wall_seconds=time.perf_counter()-self.start,
                         process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         device_peak_sampled_used_mib=max((r['used_mib'] for r in self.samples),default=None))
        if self.torch.cuda.is_available():
            self.capture_cuda_peak()
            self.meta.update(torch_peak_allocated_bytes=self.cuda_peaks['allocated_bytes'],
                             torch_peak_reserved_bytes=self.cuda_peaks['reserved_bytes'])
        if exc:
            (self.path/'logs/error.txt').write_text(''.join(traceback.format_exception(typ,exc,tb)))
            self.meta['error']=str(exc)
        write_json(self.path/'metadata.json',self.meta)
        write_json(self.path/'memory_samples.json',self.samples)
        return False
