"""Isolated HF2/Xet publisher; the pinned OpenPI Python environment stays unchanged."""
import argparse,os
os.environ['PI0_ARTIFACT_WORKER']='1'
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
# This instance has high per-connection latency. The adaptive controller settled
# at four streams after timeouts; pin bounded parallelism for artifact I/O only.
# https://huggingface.co/docs/hub/xet/using-xet-storage
os.environ.setdefault('HF_XET_FIXED_UPLOAD_CONCURRENCY','32')
os.environ.setdefault('HF_XET_FIXED_DOWNLOAD_CONCURRENCY','16')
os.environ.setdefault('HF_XET_CLIENT_READ_TIMEOUT','600s')
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
from storage import publish
p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('remote');p.add_argument('--full-readback',action='store_true');a=p.parse_args()
publish(a.folder,a.remote,full_readback=a.full_readback)
