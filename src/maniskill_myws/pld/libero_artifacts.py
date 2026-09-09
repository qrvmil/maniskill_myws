"""Immutable run directories and actual machine/runtime measurements."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import os
import resource
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import traceback


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
        import torch
        self.torch = torch
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

    def _monitor(self):
        while not self.stop_event.is_set():
            out=command_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'])
            try:
                self.samples.append(dict(seconds=time.perf_counter()-self.start,used_mib=int(out.splitlines()[0])))
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
            self.meta.update(torch_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                             torch_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        if exc:
            (self.path/'logs/error.txt').write_text(''.join(traceback.format_exception(typ,exc,tb)))
            self.meta['error']=str(exc)
        write_json(self.path/'metadata.json',self.meta)
        write_json(self.path/'memory_samples.json',self.samples)
        return False
