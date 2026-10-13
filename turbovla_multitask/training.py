"""Single-GPU training contracts; no RLDS/TensorFlow imports in the train loop."""
import numpy as np
import torch
import fcntl
from contextlib import contextmanager


@contextmanager
def training_lock(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'training.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def masked_l1(prediction, target, mask):
    mask = mask.unsqueeze(-1).to(prediction.dtype)
    return ((prediction-target).abs()*mask).sum()/(mask.sum()*prediction.shape[-1]).clamp_min(1)


@torch.no_grad()
def update_ema(ema, model, decay=.999):
    for name, parameter in model.named_parameters():
        if name in ema:
            ema[name].lerp_(parameter.detach(), 1-decay)


def sample_indices(lengths, step, batch_size=8, seed=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed,step]))
    tasks = rng.integers(len(lengths),size=batch_size)
    return [(int(task),int(rng.integers(lengths[task]))) for task in tasks]


class TaskData:
    """RAM-resident small state/action arrays, lazy compressed image reads."""
    def __init__(self, tasks, stats, work):
        import h5py
        from data import training_tasks
        self.tasks = training_tasks(tasks)
        self.handles = [h5py.File(work/'data'/f'{task}.hdf5','r') for task in tasks]
        self.episodes = []
        self.indices = []
        for handle in self.handles:
            episodes = [(handle[name],handle[name]['states'][:],handle[name]['actions'][:]) for name in sorted(handle)]
            self.episodes.append(episodes)
            self.indices.append([(ep,frame) for ep,(_,states,_) in enumerate(episodes) for frame in range(len(states))])
        self.lengths = list(map(len,self.indices))
        self.stats = stats
        self.mean = np.asarray(stats['proprio']['mean'],dtype=np.float32)
        self.std = np.asarray(stats['proprio']['std'],dtype=np.float32)
        self.image_mean = torch.tensor([.485,.456,.406]).view(1,1,3,1,1)
        self.image_std = torch.tensor([.229,.224,.225]).view(1,1,3,1,1)

    def batch(self, step, batch_size=8, seed=0):
        from data import TASKS, action_chunk, normalize_actions
        images,states,chunks,masks,instructions = [],[],[],[],[]
        for task_index,flat_frame in sample_indices(self.lengths,step,batch_size,seed):
            episode,frame = self.indices[task_index][flat_frame]
            group,s,a = self.episodes[task_index][episode]
            images.append(group['images'][frame])
            states.append((s[frame]-self.mean)/(self.std+1e-6))
            chunk,mask = action_chunk(a,frame)
            chunks.append(normalize_actions(chunk,self.stats))
            masks.append(mask)
            instructions.append(TASKS[self.tasks[task_index]]['name'].replace('_',' '))
        pixels = torch.from_numpy(np.stack(images)).permute(0,1,4,2,3).float()/255
        pixels = (pixels-self.image_mean)/self.image_std
        return pixels,instructions,torch.from_numpy(np.stack(states)),torch.from_numpy(np.stack(chunks)),torch.from_numpy(np.stack(masks))

    def close(self):
        for handle in self.handles:
            handle.close()
