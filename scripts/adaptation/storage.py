"""Publish progress, verify remote bytes, preserve resumable model/optimizer states."""
import json, os, subprocess, time
from pathlib import Path
from common import ROOT,WORK,HF_REPO,EXPERIMENT,sha,write

def publish(folder,remote,*,full_readback=False,defer=False):
 if defer:
  from publication_queue import submit
  return submit(folder,remote,full_readback)
 if os.environ.get('PI0_ARTIFACT_WORKER')!='1':
  environment=dict(os.environ,PI0_ARTIFACT_WORKER='1',HF_XET_HIGH_PERFORMANCE='1',CUDA_VISIBLE_DEVICES='')
  environment.pop('HF_HUB_ENABLE_HF_TRANSFER',None)
  command=['/venv/main/bin/python','-u',str(ROOT/'scripts/adaptation/publish_remote.py'),str(Path(folder).resolve()),remote]
  if full_readback:command.append('--full-readback')
  subprocess.run(command,cwd=ROOT,env=environment,check=True)
  return json.loads((Path(folder)/'upload_receipt.json').read_text())
 from huggingface_hub import HfApi,hf_hub_download,hf_hub_url
 folder=Path(folder)
 inventory={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('upload_receipt.json','inventory.json')}
 write(folder/'inventory.json',inventory)
 api=HfApi();started=time.time()
 commit=api.upload_folder(repo_id=HF_REPO,folder_path=folder,path_in_repo=remote,commit_message='Save '+remote,ignore_patterns=['upload_receipt.json'])
 uploaded=time.time()
 paths=[remote+'/'+name for name in inventory]
 infos={}
 for offset in range(0,len(paths),200):
  infos.update({item.path:item for item in api.get_paths_info(HF_REPO,paths[offset:offset+200],revision=commit.oid)})
 # Independent readbacks share one batched metadata request and bounded network concurrency.
 def verify(item):
  name,meta=item;path=remote+'/'+name;info=infos[path]
  if info.lfs:
   checksum=info.lfs.sha256 if hasattr(info.lfs,'sha256') else info.lfs['sha256']
   assert checksum==meta['sha256'],path
  assert info.size==meta['bytes'],path
  if meta['bytes']<10_000_000 or (full_readback and name!='training_state.pt'):
   tmp=WORK/'readback'/remote
   dest=Path(hf_hub_download(HF_REPO,path,revision=commit.oid,local_dir=tmp,force_download=True))
   assert sha(dest)==meta['sha256'],path
   if meta['bytes']>=10_000_000:dest.unlink()
   method='download+sha256'
  else:
   url=hf_hub_url(HF_REPO,path,revision=commit.oid)
   for start in (0,max(0,meta['bytes']-65536)):
    end=min(meta['bytes']-1,start+65535)
    import requests
    from huggingface_hub.utils import build_hf_headers
    headers=build_hf_headers();headers['Range']=f'bytes={start}-{end}'
    with requests.get(url,headers=headers,stream=True,timeout=120) as response:
     response.raise_for_status();assert response.status_code==206,(path,response.status_code)
     data=response.content
    with open(folder/name,'rb') as stream:
     stream.seek(start);assert data==stream.read(end-start+1)
   method='remote LFS sha256 + first/last 64KiB readback'
  return dict(path=path,verification=method,sha256=meta['sha256'])
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=4) as executor:verified=list(executor.map(verify,inventory.items()))
 receipt=dict(repo=HF_REPO,commit=commit.oid,verified=verified,time=time.time(),upload_seconds=uploaded-started,verification_seconds=time.time()-uploaded)
 write(folder/'upload_receipt.json',receipt)
 evidence=EXPERIMENT/'evidence'/ (remote.replace('/','_')+'.json')
 write(evidence,receipt)
 print('HF_VERIFIED',remote,commit.oid,flush=True)
 return receipt

def _push_progress_unlocked(message):
 dest=EXPERIMENT
 import datetime
 lines=['# Статус эксперимента','', 'Обновлено UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat(), '', 'GitHub/HF write/read probes прошли. Бюджеты и правила выбора: [PREREGISTRATION.md](PREREGISTRATION.md).', '']
 for stage in ('d0','full','expert'):
  p=WORK/'sft'/stage/'latest.json'
  latest=json.loads(p.read_text()) if p.exists() else None
  metrics=WORK/'sft'/stage/'metrics.jsonl';live=None
  if metrics.exists():
   try:live=json.loads(metrics.read_text().strip().splitlines()[-1])['step']
   except (ValueError,IndexError):pass
  saved=str(latest['step'])+' updates; checkpoint '+latest['checkpoint'] if latest else 'ещё нет сохранённого checkpoint'
  lines.append('- SFT '+stage+': '+(str(live)+' выполненных updates; ' if live is not None else '')+saved)
 p=WORK/'teacher.json'
 lines.append('- Teacher: '+(str(json.loads(p.read_text())['step'])+' D1 full-SFT updates, validation passed' if p.exists() else 'ещё не выбран по отдельной validation'))
 p=WORK/'rl/latest.json'
 lines.append('- Residual RL: '+(str(json.loads(p.read_text())['counters']) if p.exists() else 'ещё нет сохранённого checkpoint'))
 final_rows=list((WORK/'final').glob('*/*/*/episode.json'))
 lines+=['- Финальные episodes: '+str(len(final_rows))+'/300. Каждому завершённому episode соответствует проверенное двухкамерное видео с terminal frame.', '', 'Final seeds не используются для настройки или выбора checkpoints. Завершённые episodes при resume проверяются и переиспользуются без увеличения N.', '', 'Код: `/workspace/pi0_experiment`; артефакты: `/workspace/pi0_three`; среда: `/workspace/pi0_venv`. Supervisor и логи: `pi0_three_d0`, `pi0_three_pipeline`, `/var/log/portal/<name>.log`.']
 if (dest/'results/completeness.json').exists():lines+=['', '[Результаты и completeness check](results/completeness.json); [краткий отчёт](REPORT.md).']
 if os.environ.get('PI0_VISUAL_ALIGNMENT')=='1':
  lines=[line for line in lines if not line.startswith('Код: `/workspace/pi0_experiment`')]
  lines += ['', 'Новый запуск: alignment общей D0-базы; full D1 продолжает alignment; expert и residual сохраняют исходные frozen scopes и наследуют aligned D0. Протокол: PREREGISTRATION.md.', f'Рабочий код: `{ROOT}`; артефакты: `{WORK}`; HF: https://huggingface.co/{HF_REPO}.', 'Аппаратный и модельный preflight прошли; исходные historical results не являются результатами этого запуска.']
 (dest/'STATUS.md').write_text('\n'.join(lines)+'\n')
 for src in (WORK/'evidence').iterdir():
  if src.suffix not in ('.json','.png'):continue
  import shutil
  (dest/'evidence').mkdir(exist_ok=True);shutil.copy2(src,dest/'evidence'/src.name)
 subprocess.run(['git','add','scripts/adaptation','experiments/pi0_d0_d1_three_adaptations'],cwd=ROOT,check=True)
 subprocess.run(['git','add','-f',*map(str,dest.rglob('*.json'))],cwd=ROOT,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
  subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
  subprocess.run(['git','push','origin','HEAD'],cwd=ROOT,check=True)


def push_progress(message):
 import fcntl
 WORK.mkdir(exist_ok=True)
 with (WORK/'git_progress.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  _push_progress_unlocked(message)
