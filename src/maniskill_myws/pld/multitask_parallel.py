"""Parallelize independent simulator/policy processes only after exact replay validation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from .multitask_data import WORK,ROOT
from .multitask_protocol import TASKS,sha256
from .libero_artifacts import write_json


def execution_signature():
    import importlib.metadata
    import hashlib
    sources={p.name:sha256(p) for p in (ROOT/'src/maniskill_myws/pld').glob('*.py')}
    packages={p:importlib.metadata.version(p) for p in
              ('jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','torch','numpy','mujoco','robosuite','flax','orbax-checkpoint')}
    dependencies={}
    for name,path in [('openpi',ROOT/'third_party/openpi'),('LIBERO',Path('/workspace/LIBERO'))]:
        dependencies[name]={
            'revision':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            'diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(path),'diff','HEAD'])).hexdigest()}
    checkpoint=WORK/'A/checkpoints/pi0_libero_seen_lora32/EXP-001/0'
    identity=[(str(p.relative_to(checkpoint)),p.stat().st_size,p.stat().st_mtime_ns)
              for p in sorted(checkpoint.rglob('*')) if p.is_file()]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    return dict(sources=sources,packages=packages,dependencies=dependencies,gpu=gpu,
        config=sha256(ROOT/'configs/pld_libero/d0_base_d1_residual.json'),
        checkpoint=str(checkpoint),checkpoint_files=identity,
        environment={k:os.environ.get(k) for k in ('JAX_PLATFORMS','XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE','MUJOCO_GL')})


def compare_rollouts(reference,other):
    if len(reference)!=len(other):raise ValueError('Parallel validation episode count differs')
    keys=('seed','success','length','reset_hash','trajectory_hash','image_hash')
    for a,b in zip(reference,other):
        for key in keys:
            if a[key]!=b[key]:raise ValueError(f'Parallel evaluation changed {key}')


def _run(arguments,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/eval_multitask_sft.py'),*arguments,'--serial'],
                       cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def validate_parallel():
    root=WORK/'parallel_validation';root.mkdir(parents=True,exist_ok=True)
    checkpoint=WORK/'A/checkpoints/pi0_libero_seen_lora32/EXP-001/0'
    if not checkpoint.exists():return False
    signature=execution_signature()
    # Normalize tuples to their serialized representation for exact cache comparison.
    signature=json.loads(json.dumps(signature))
    existing=root/'decision.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old['source_signature']!=signature:
            print('PARALLEL_CACHE_STALE: execution identity changed; using serial',flush=True)
            write_json(root/'last_fallback.json',dict(reason='Execution identity changed',source_signature=signature))
            return False
        return old['safe']
    common=['--checkpoint',str(checkpoint),'--variant','A','--task','D0','--episodes','2','--videos','0']
    safe=False;reason=''
    try:
        reference=root/'reference'
        _run([*common,'--output',str(reference),'--reset-dir',str(reference/'resets')],root/'reference.log')
        def worker(index):
            out=root/f'concurrent_{index}'
            _run([*common,'--output',str(out),'--reset-dir',str(out/'resets')],root/f'concurrent_{index}.log')
            return json.loads((out/'D0/episodes.json').read_text())
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(worker,range(3)))
        ref=json.loads((reference/'D0/episodes.json').read_text())
        for result in rows:compare_rollouts(ref,result)
        safe=True;reason='Three concurrent independent policy/simulator processes exactly match serial outcomes, lengths, reset, trajectory and image hashes on two D0 seeds.'
    except (subprocess.CalledProcessError,ValueError) as error:
        reason=f'Validation failed; serial fallback: {error}'
    write_json(existing,dict(safe=safe,reason=reason,source_signature=signature,
        task='D0',variant='A',updates=0,seeds=[10000,10001],workers=3,
        checkpoint=str(checkpoint),uses_heldout_outcomes=False))
    print('PARALLEL_VALIDATION',safe,reason,flush=True)
    return safe


def dispatch(args):
    """Return true when completed here; false asks caller to use its serial main."""
    if args.serial or args.sensitivity_only:return False
    tasks=list(TASKS) if args.task=='all' else args.task.split(',')
    if len(tasks)<2 or not all(t in TASKS for t in tasks):return False
    if not validate_parallel():return False
    from .multitask_protocol import TRAIN_SETS
    from .multitask_data import repo_id
    variant=args.variant
    if variant is None:
        matches=[v for v in TRAIN_SETS if (Path(args.checkpoint)/'assets'/repo_id(v)/'norm_stats.json').exists()]
        if len(matches)!=1:raise ValueError('Supply compatible variant')
        variant=matches[0]
    output=Path(args.output) if args.output else WORK/'eval'/variant/Path(args.checkpoint).name
    output.mkdir(parents=True,exist_ok=True)
    common=['--checkpoint',args.checkpoint,'--variant',variant,'--episodes',str(args.episodes),
            '--seed-start',str(args.seed_start),'--videos',str(args.videos),'--output',str(output)]
    if args.seeds:common+=['--seeds',args.seeds]
    if args.normalization:common+=['--normalization',args.normalization]
    if args.reset_dir:common+=['--reset-dir',args.reset_dir]
    def worker(task):
        _run([*common,'--task',task,'--runtime-label','runtime_'+task],output/f'worker_{task}.log')
        print('PARALLEL_TASK_COMPLETE',variant,task,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,tasks))
    if args.image_sensitivity:
        _run([*common,'--task','D0','--runtime-label','runtime_sensitivity','--sensitivity-only','--image-sensitivity'],
             output/'sensitivity.log')
    write_json(output/'parallel_dispatch.json',dict(workers=3,tasks=tasks,validation_sha256=sha256(WORK/'parallel_validation/decision.json')))
    return True
