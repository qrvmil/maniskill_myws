"""Preregistered multi-task scientific contracts; no GPU or simulator imports."""
from pathlib import Path
import numpy as np
from .libero_protocol import file_sha256 as sha256
from .libero_sanity import wilson

BASE_URI='gs://openpi-assets/checkpoints/pi0_base'
BASE_COMMIT='c9b14ef43f7a94213536ebb97f3056bfd9f9dd87'
OPENPI_REV='981483dca0fd9acba698fea00aa6e52d56a66c58'
LIBERO_REV='8f1084e3132a39270c3a13ebe37270a43ece2a01'
TRAIN_SETS={'A':('D0',),'B':('D0','D1'),'C':('D0','D1','D2')}
UPDATES=(0,500,1000,2000,3001)
SEEDS=tuple(range(10000,10050))
TASKS={
    'D0':dict(suite='libero_spatial',name='pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',horizon=220),
    'D1':dict(suite='libero_spatial',name='pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate',horizon=220),
    'D2':dict(suite='libero_goal',name='put_the_bowl_on_the_stove',horizon=280),
    'H1':dict(suite='libero_spatial',name='pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate',horizon=220),
    'H2':dict(suite='libero_object',name='pick_up_the_alphabet_soup_and_place_it_in_the_basket',horizon=280),
}
BDDL_HASHES={'D0': '5e19880ebc844f86b89f63b172aab0b7a89f85b6e6955891f544f0ea7d0795d5', 'D1': 'c8fbb9effa27b947d04d26862d75273d93cac1806734e53836cd472a08aec1fe', 'D2': 'c6812d6c988be830412aa6fc4801af04d60f8c4565d2625763b6a0f2b843a87f', 'H1': '53a7516571412a2f46a27cbf8482d3b76dbad4221858c8f6b565d506c274e61d', 'H2': 'df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e'}
for _key,_task in TASKS.items():
    _task['distance']=_key
    _task['bddl_sha256']=BDDL_HASHES[_key]


def prompt(task):return TASKS[task]['name'].replace('_',' ')


def require_training_tasks(variant,tasks):
    if tuple(tasks)!=TRAIN_SETS[variant]:raise ValueError('Training corpus differs or contains held-out leakage')


class BalancedSampler:
    """Uniform task draws; shuffled cycling examples within each task.

    Task probabilities do not depend on frame or demonstration counts. A finite
    stream is long enough for all optimizer batches plus the upstream lookahead.
    """
    def __init__(self,lengths,num_samples,seed=0):
        self.lengths=np.asarray(lengths,dtype=int)
        if self.lengths.ndim!=1 or len(self.lengths)==0 or np.any(self.lengths<=0) or num_samples<1:
            raise ValueError('Nonempty tasks and positive sample count required')
        self.num_samples=int(num_samples);self.seed=seed;self.task_log=[];self.index_log=[]

    def __len__(self):return self.num_samples

    def __iter__(self):
        rng=np.random.default_rng(self.seed)
        offsets=np.r_[0,np.cumsum(self.lengths)[:-1]]
        queues=[rng.permutation(n).tolist() for n in self.lengths]
        self.task_log=[];self.index_log=[]
        for _ in range(self.num_samples):
            t=int(rng.integers(len(self.lengths)))
            if not queues[t]:queues[t]=rng.permutation(self.lengths[t]).tolist()
            index=int(offsets[t]+queues[t].pop())
            self.task_log.append(t);self.index_log.append(index)
            yield index


def require_binding(binding,variant,base_manifest_hash):
    require_training_tasks(variant,binding['training_tasks'])
    if (binding['variant']!=variant or binding['normalization_training_tasks']!=list(TRAIN_SETS[variant])
            or binding['demo_counts']!={t:50 for t in TRAIN_SETS[variant]}
            or binding['base_uri']!=BASE_URI or binding['base_manifest_sha256']!=base_manifest_hash):
        raise ValueError('Normalization corpus or pretrained initialization provenance mismatch')
    if sha256(binding['normalization_path'])!=binding['normalization_sha256']:
        raise ValueError('Normalization hash mismatch')


def summarize(rows):
    if not rows or len({r['seed'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate episodes')
    n=len(rows);s=sum(bool(r['success']) for r in rows)
    return dict(n=n,successes=s,success_rate=s/n,wilson_95ci=wilson(s,n),
                mean_episode_length=float(np.mean([r['length'] for r in rows])))


def paired_change(before,after):
    summarize(before);summarize(after)
    a=sorted(before,key=lambda r:r['seed']);b=sorted(after,key=lambda r:r['seed'])
    if len(a)!=len(b) or any((x['seed'],x['reset_hash'])!=(y['seed'],y['reset_hash']) for x,y in zip(a,b)):
        raise ValueError('Paired seeds/reset states differ')
    delta=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
    boot=np.random.default_rng(0).choice(delta,size=(10000,len(delta))).mean(1)
    ci=(100*np.quantile(boot,[.025,.975])).tolist()
    # Report requested bootstrap honestly, including boundary degeneracy, alongside an exact bound.
    boundary=None
    if np.all(delta==0):
        bound=100*(1-.05**(1/len(delta)));boundary=[-bound,bound]
    elif np.all(delta==1) or np.all(delta==-1):
        bound=100*(2*.05**(1/len(delta))-1)
        boundary=[bound,100.] if delta[0]==1 else [-100.,-bound]
    return dict(n=len(delta),rescue=int(sum(delta>0)),harm=int(sum(delta<0)),gain_pp=float(100*delta.mean()),
                bootstrap_95_pp=ci,bootstrap_draws=10000,bootstrap_seed=0,exact_boundary_95_pp=boundary)


def select_videos(rows,count=2):
    if count<0:raise ValueError('Video count cannot be negative')
    counts={False:0,True:0};out=[]
    for row in sorted(rows,key=lambda r:r['seed']):
        outcome=bool(row['success'])
        if counts[outcome]<count:out.append(row);counts[outcome]+=1
    return out


def video_name(variant,task,row):
    if variant=='external':
        return f'external_{task}_seed{row["seed"]}_{"success" if row["success"] else "failure"}.mp4'
    return f'train_{"_".join(TRAIN_SETS[variant])}_{task}_seed{row["seed"]}_{"success" if row["success"] else "failure"}.mp4'


def validate_video_metadata(meta,row,variant,task):
    for key in ('seed','success','length','reset_hash','trajectory_hash','image_hash'):
        if meta[key]!=row[key]:raise ValueError(f'Video metadata differs: {key}')
    if (meta['variant']!=variant or meta['task']!=task or not meta['includes_terminal_frame']
            or meta['frames']!=row['length']+1 or meta['expected_frames']!=row['length']+1):
        raise ValueError('Video model/task/frame metadata differs')


def action_changes(correct,shuffled):
    a=np.asarray(correct);b=np.asarray(shuffled)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or len(a)<5 or not np.isfinite([a,b]).all():
        raise ValueError('Expected matching finite LIBERO action chunks [H>=5,7]')
    dist=np.linalg.norm(a-b,axis=1)
    return dict(first_action_l2=float(dist[0]),first5_mean_l2=float(dist[:5].mean()),chunk_mean_l2=float(dist.mean()))
