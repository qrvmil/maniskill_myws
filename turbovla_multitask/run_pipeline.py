"""Managed sequential A/B/C training, verified HF publication and15-episode eval."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('HF_HOME','/workspace/.hf_home')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from data import ROOT,WORK,TRAIN_SETS,TASKS,HF_REPO
from artifacts import verified_final
from prepare import sha256,write_json
from report import summarize_cell

BRANCH='exp/turbovla-multitask-generalization'


def command(argv):
    print('COMMAND',argv,flush=True)
    subprocess.run(argv,cwd=ROOT,check=True)


def script(name,*args):
    command([sys.executable,str(ROOT/'turbovla_multitask'/name),*args])


def record(phase,variant=None,**extra):
    state=dict(phase=phase,variant=variant,pid=os.getpid(),**extra)
    write_json(WORK/'pipeline_state.json',state)
    print('PIPELINE_STATE',json.dumps(state),flush=True)


def wait_existing_a():
    """Only supervisor's authoritative state decides when its existing job ended."""
    while True:
        result=subprocess.run(['supervisorctl','status','turbovla_train_a'],text=True,capture_output=True)
        words=result.stdout.split()
        if len(words)>=2 and words[1] in ('RUNNING','STARTING','BACKOFF','STOPPING'):
            record('waiting_existing_training','A',supervisor=result.stdout.strip())
            time.sleep(15)
            continue
        if len(words)>=2 and words[1] in ('EXITED','STOPPED','FATAL'):
            print('EXISTING_A_TERMINAL',result.stdout.strip(),flush=True)
            return
        # A fresh machine may have no previous managed job. Fail on observation
        # errors instead of assuming an unobserved process is terminal.
        if 'no such process' in result.stdout.lower():
            return
        raise RuntimeError(f'Cannot establish existing A process state: {result.stdout} {result.stderr}')


def commit_files(paths,message):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH: raise ValueError('Experiment branch changed; preserve work and stop')
    staged=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
    if staged.returncode!=0: raise ValueError('Existing staged changes: do not commit unrelated work')
    command(['git','add','-f',*[str(Path(p).relative_to(ROOT)) for p in paths]])
    result=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
    if result.returncode==1:
        command(['git','diff','--cached','--check'])
        command(['git','commit','-m',message])
    elif result.returncode!=0: raise RuntimeError('Cannot inspect staged evidence')
    command(['git','push','origin',BRANCH])


def publish_evaluation(variant,output):
    from huggingface_hub import HfApi,hf_hub_download
    # Videos and episode evidence survive loss of the nonpersistent instance.
    hashes={str(path.relative_to(output)):sha256(path) for path in output.rglob('*') if path.is_file()}
    info=HfApi().upload_folder(repo_id=HF_REPO,folder_path=output,path_in_repo=f'{variant}/evaluation',
                              commit_message=f'Archive TurboVLA {variant}75 reference-protocol episodes')
    for name,expected in hashes.items():
        cached=hf_hub_download(HF_REPO,f'{variant}/evaluation/{name}',revision=info.oid)
        if sha256(cached)!=expected: raise ValueError(f'Remote evaluation evidence differs: {name}')
    publication=dict(repo=HF_REPO,revision=info.oid,variant=variant,hashes=hashes,verified_download=True)
    write_json(WORK/variant/'evaluation_hf_publication.json',publication)
    return publication


def pipeline():
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_existing_a()
        for variant in TRAIN_SETS:
            record('training',variant)
            if not (WORK/variant/'train'/'complete.json').exists():
                script('train.py','--variant',variant)
            checkpoint,stats,metadata=verified_final(variant)
            record('publishing_weights',variant,checkpoint_sha256=metadata['checkpoint_sha256'])
            script('publish.py','--variant',variant)
            evidence=ROOT/'docs'/'turbovla_multitask'/'variants'/variant
            evidence.mkdir(parents=True,exist_ok=True)
            paths=[]
            for source,name in [(WORK/variant/'train'/'complete.json','training_complete.json'),
                                (WORK/variant/'train'/'config.json','training_config.json'),
                                (WORK/variant/'train'/'train.jsonl','train.jsonl'),(stats,'stats.json')]:
                target=evidence/name;shutil.copy2(source,target);paths.append(target)
            paths.append(ROOT/'docs'/'turbovla_multitask'/'publications'/f'{variant}.json')
            commit_files(paths,f'results: publish verified TurboVLA {variant} final weights and training evidence')
            record('evaluating',variant)
            output=WORK/variant/'eval15'
            if not (output/'complete.json').exists():
                if output.exists():
                    raise FileExistsError(f'Incomplete previous evaluation: inspect {output}; do not overwrite or duplicate')
                script('evaluate.py','--variant',variant)
            config=json.loads((output/'config.json').read_text())
            complete=json.loads((output/'complete.json').read_text())
            if (config['checkpoint_sha256']!=metadata['checkpoint_sha256'] or config['stats_sha256']!=metadata['stats_sha256']
                or config['tasks']!=list(TASKS) or config['episodes']!=15 or complete!={'total_episodes':75,'config':config}):
                raise ValueError('Evaluation completion/identity differs')
            summaries={}
            for task in TASKS:
                summaries[task]=summarize_cell(json.loads((output/task/'episodes.json').read_text()),variant,task)
            record('archiving_evaluation',variant)
            publication=publish_evaluation(variant,output)
            paths=[]
            for source in [output/'config.json',output/'complete.json',*[output/task/'episodes.json' for task in TASKS]]:
                target=evidence/'evaluation'/source.relative_to(output)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);paths.append(target)
            summary_path=evidence/'evaluation'/'recomputed_summary.json';write_json(summary_path,summaries);paths.append(summary_path)
            pub_path=ROOT/'docs'/'turbovla_multitask'/'publications'/f'{variant}_evaluation.json'
            write_json(pub_path,publication);paths.append(pub_path)
            commit_files(paths,f'results: archive and verify all75 TurboVLA {variant} evaluation episodes')
        record('building_matrix')
        script('report.py')
        script('build_notebook.py')
        paths=[ROOT/'docs/turbovla_multitask'/name for name in
               ['results.json','generalization_matrix.csv','generalization_matrix.png','RESULTS.md']]
        paths.append(ROOT/'notebooks/02_turbovla_multitask_eval.ipynb')
        commit_files(paths,'results: add complete TurboVLA3×5 generalization matrix')
        record('models_and_matrix_complete',notebook_execution_pending=True)


if __name__=='__main__':pipeline()
