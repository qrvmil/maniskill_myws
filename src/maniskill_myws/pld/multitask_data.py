"""Own-corpus conversion and normalization using pinned OpenPI/LeRobot APIs."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
from .multitask_protocol import TRAIN_SETS,TASKS,BASE_URI,OPENPI_REV,LIBERO_REV,prompt,sha256,require_training_tasks
from .libero_artifacts import write_json
from .libero_protocol import directory_manifest,verify_directory
from .libero_alignment import native_observation_action_indices

ROOT=Path(__file__).resolve().parents[3]
WORK=Path(os.environ.get('MULTITASK_WORK','/workspace/multitask-sft'))
DATA=WORK/'lerobot'
BASE=WORK/'base/pi0_base'
V4=json.loads((ROOT/'configs/pld_libero/d0_base_d1_residual.json').read_text())


def repo_id(variant):return f'local/multitask_{variant}'


def train_config(variant):
    from .libero_alignment import make_openpi_config
    cfg=make_openpi_config(V4,repo_id=repo_id(variant),workdir=WORK/variant,method='lora32',steps=3001)
    return dataclasses.replace(cfg,save_interval=1,keep_period=1,
        weight_loader=dataclasses.replace(cfg.weight_loader,params_path=str(BASE/'params')))


def load_script(name):
    spec=importlib.util.spec_from_file_location('multitask_openpi_'+name,ROOT/'third_party/openpi/scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def audit_h5(path,task):
    import h5py
    if task not in ('D0','D1','D2'):raise ValueError('Held-out demonstrations forbidden')
    spec=TASKS[task]
    with h5py.File(path,'r') as f:
        data=f['data'];bddl=str(data.attrs.get('bddl_file_name',''))
        if not bddl.endswith(f"{spec['suite']}/{spec['name']}.bddl"):
            raise ValueError(f'Wrong demonstration BDDL: {bddl}')
        demos=sorted(data,key=lambda x:int(x.split('_')[-1]))
        if len(demos)!=50:raise ValueError('Primary corpus requires 50 demos per task')
        lengths=[]
        for demo in demos:
            g=data[demo];a=g['actions'][:];o=g['obs']
            if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all() or np.max(np.abs(a))>1:
                raise ValueError('Invalid 7D bounded OSC actions')
            if any(len(o[k])!=len(a) for k in ('agentview_rgb','eye_in_hand_rgb','ee_pos','ee_ori','gripper_states')):
                raise ValueError('Observation/action length mismatch')
            lengths.append(len(a)-1)
    return dict(task=task,path=str(path),sha256=sha256(path),bddl=bddl,prompt=prompt(task),
                demo_names=demos,aligned_lengths=lengths,frames=sum(lengths),num_demonstrations=50)


def fetch():
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    for folder,rev in [(ROOT/'third_party/openpi',OPENPI_REV),(Path('/workspace/LIBERO'),LIBERO_REV)]:
        if subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()!=rev:
            raise ValueError('Dependency revision differs')
    (WORK/'native').mkdir(parents=True,exist_ok=True)
    def one(task):
        t=TASKS[task];path=WORK/'native'/f'{task}.hdf5'
        url=f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/{t['suite']}/{t['name']}_demo.hdf5"
        if not path.exists():
            temporary=path.with_suffix('.download');urllib.request.urlretrieve(url,temporary)
            audit_h5(temporary,task);temporary.replace(path)
        audit=audit_h5(path,task);audit['download_url']=url
        write_json(path.with_suffix('.json'),audit);print('SOURCE_VERIFIED',task,audit['frames'],flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,('D0','D1','D2')))
    # Use the official storage inventory to verify every downloaded object, including byte sizes.
    import requests
    prefix='checkpoints/pi0_base/'
    objects=[];page=None
    while True:
        params={'prefix':prefix}
        if page:params['pageToken']=page
        response=requests.get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o',params=params,timeout=60)
        response.raise_for_status();inventory=response.json();objects.extend(inventory.get('items',[]))
        page=inventory.get('nextPageToken')
        if not page:break
    if not objects:raise RuntimeError('Empty official checkpoint inventory')
    BASE.mkdir(parents=True,exist_ok=True)
    def download(obj):
        relative=obj['name'][len(prefix):]
        if not relative or obj['name'].endswith('/'):return
        path=BASE/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or path.stat().st_size!=int(obj['size']):
            tmp=path.with_name(path.name+'.download')
            with requests.get('https://storage.googleapis.com/openpi-assets/'+obj['name'],stream=True,timeout=120) as r:
                r.raise_for_status()
                with tmp.open('wb') as out:
                    for block in r.iter_content(8*1024*1024):out.write(block)
            if tmp.stat().st_size!=int(obj['size']):raise RuntimeError('Checkpoint byte size mismatch')
            tmp.replace(path)
        import base64,hashlib
        digest=hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if base64.b64encode(digest.digest()).decode()!=obj['md5Hash']:raise ValueError('Official storage checksum mismatch')
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(download,objects))
    write_json(WORK/'base_inventory.json',objects)
    write_json(WORK/'base_manifest.json',directory_manifest(BASE))
    print('OFFICIAL_BASE_VERIFIED',len(objects),flush=True)


def convert(variant):
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    tasks=TRAIN_SETS[variant];require_training_tasks(variant,tasks)
    root=DATA/repo_id(variant);audit_path=root/'source_audit.json'
    if audit_path.exists():
        cached=json.loads(audit_path.read_text())
        if cached['variant']!=variant or cached['repo_id']!=repo_id(variant):
            raise ValueError('Cached dataset variant differs')
        require_training_tasks(variant,cached['training_tasks'])
        for t in tasks:
            if cached['sources'][t]!=audit_h5(WORK/'native'/f'{t}.hdf5',t):
                raise ValueError('Cached source corpus differs')
        verify_directory(root,cached['dataset_files'],exclude=('source_audit.json',))
        return cached
    sources={t:audit_h5(WORK/'native'/f'{t}.hdf5',t) for t in tasks}
    shape=(128,128,3)
    ds=LeRobotDataset.create(repo_id=repo_id(variant),root=root,robot_type='panda',fps=20,
        features={'image':{'dtype':'image','shape':shape,'names':['height','width','channel']},
                  'wrist_image':{'dtype':'image','shape':shape,'names':['height','width','channel']},
                  'state':{'dtype':'float32','shape':(8,),'names':['state']},
                  'actions':{'dtype':'float32','shape':(7,),'names':['actions']}},
        image_writer_threads=4,image_writer_processes=0)
    for t in tasks:
        with h5py.File(sources[t]['path'],'r') as f:
            for demo in sources[t]['demo_names']:
                g=f['data'][demo];o=g['obs']
                for i,j in native_observation_action_indices(len(g['actions'])):
                    state=np.concatenate([o['ee_pos'][i],o['ee_ori'][i],o['gripper_states'][i]]).astype(np.float32)
                    ds.add_frame({'image':np.ascontiguousarray(o['agentview_rgb'][i][::-1,::-1]),
                        'wrist_image':np.ascontiguousarray(o['eye_in_hand_rgb'][i][::-1,::-1]),
                        'state':state,'actions':np.asarray(g['actions'][j],np.float32),'task':prompt(t)})
                ds.save_episode()
        print('CONVERTED',variant,t,flush=True)
    ds.stop_image_writer()
    audit=dict(variant=variant,training_tasks=list(tasks),sources=sources,repo_id=repo_id(variant),
               task_frame_counts=[sources[t]['frames'] for t in tasks],
               camera_transform='rotate180 both cameras; official resize_with_pad224',
               alignment='native obs[i] -> action[i+1]; drop terminal observation',
               dataset_files=directory_manifest(root,exclude=('source_audit.json',)))
    write_json(audit_path,audit);return audit


def validate_cached_provenance(existing,expected):
    if existing!=expected:raise ValueError('Existing normalization provenance differs or is missing')


def prepare(variant):
    os.environ['HF_LEROBOT_HOME']=str(DATA)
    audit=convert(variant);cfg=train_config(variant)
    from openpi.training import config as oc
    oc._CONFIGS_DICT[cfg.name]=cfg
    norm=cfg.assets_dirs/repo_id(variant)/'norm_stats.json'
    existed=norm.exists()
    if not existed:load_script('compute_norm_stats').main(cfg.name)
    binding=dict(variant=variant,training_tasks=list(TRAIN_SETS[variant]),
        demo_counts={t:50 for t in TRAIN_SETS[variant]},normalization_path=str(norm),
        normalization_sha256=sha256(norm),normalization_training_tasks=list(TRAIN_SETS[variant]),
        normalization_method='official OpenPI chunk-weighted mean/std; complete batches, batch8',
        normalization_frames_used=sum(audit['task_frame_counts'])//8*8,
        base_uri=BASE_URI,base_manifest_sha256=sha256(WORK/'base_manifest.json'),
        source_audit_sha256=sha256(DATA/repo_id(variant)/'source_audit.json'),
        task_frame_counts=audit['task_frame_counts'])
    provenance=norm.with_name('normalization_provenance.json')
    if existed:
        validate_cached_provenance(json.loads(provenance.read_text()) if provenance.exists() else None,binding)
    write_json(WORK/variant/'binding.json',binding)
    if not existed:write_json(provenance,binding)
    (WORK/variant/'train_config.txt').write_text(repr(cfg))
    print('NORMALIZATION_BOUND',variant,binding['normalization_sha256'],flush=True)
