"""Readable TurboVLA/LIBERO experiment helpers. Neural and simulator imports are lazy."""
from pathlib import Path
from collections import deque
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TURBO_ROOT = ROOT.parent / 'TurboVLA'
LIBERO_ROOT = ROOT / 'LIBERO'
ASSETS = ROOT / 'assets'
TASKS = {
    'D0': dict(suite='libero_spatial', name='pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate', horizon=220),
    'D1': dict(suite='libero_spatial', name='pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate', horizon=220),
    'D2': dict(suite='libero_goal', name='put_the_bowl_on_the_stove', horizon=280),
    'H1': dict(suite='libero_spatial', name='pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate', horizon=220),
    'H2': dict(suite='libero_object', name='pick_up_the_alphabet_soup_and_place_it_in_the_basket', horizon=280),
}
BDDL_HASHES = {'D0':'5e19880ebc844f86b89f63b172aab0b7a89f85b6e6955891f544f0ea7d0795d5', 'D1':'c8fbb9effa27b947d04d26862d75273d93cac1806734e53836cd472a08aec1fe', 'D2':'c6812d6c988be830412aa6fc4801af04d60f8c4565d2625763b6a0f2b843a87f', 'H1':'53a7516571412a2f46a27cbf8482d3b76dbad4221858c8f6b565d506c274e61d', 'H2':'df088984da13131f8332ee0f13a7896c6a97afd02ee5007a42e8fc5e0893571e'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def configure():
    """Set rendering and offline model paths BEFORE importing simulators/transformers."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'libero_config'))
    for path in [TURBO_ROOT, LIBERO_ROOT]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
    config_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    benchmark_root = LIBERO_ROOT / 'libero/libero'
    paths = {'benchmark_root': benchmark_root, 'bddl_files': benchmark_root / 'bddl_files',
             'init_states': benchmark_root / 'init_files', 'datasets': ROOT / 'datasets',
             'assets': benchmark_root / 'assets'}
    (ROOT / 'datasets').mkdir(exist_ok=True)
    (config_dir / 'config.yaml').write_text(yaml.safe_dump({k: str(v) for k,v in paths.items()}))


def load_policy(precision='bf16'):
    configure()
    import torch
    from turbovla.evaluation.suite_policy import TurboVLAPolicy
    torch.set_num_threads(2)
    return TurboVLAPolicy(ckpt_path=str(ASSETS / 'turbovla_libero.pth'),
        dinov3_path=str(ASSETS / 'dinov3'), bert_path=str(ASSETS / 'bert'),
        stats_path=str(ASSETS / 'libero_all4_stats.json'), stats_key='libero_all4_no_noops',
        precision=precision, allow_hf_download=False)


def find_task(suite, name):
    matches = [(i,task) for i,task in enumerate(suite.tasks) if task.name == name]
    if len(matches) != 1:
        raise ValueError(f'Task name must resolve uniquely: {name}; matches={len(matches)}')
    return matches[0]


def list_tasks(suite_name):
    configure()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    return [dict(task_id=i, name=t.name, language=t.language) for i,t in enumerate(suite.tasks)]


def make_env(task_key, protocol='reference'):
    configure()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    spec = TASKS[task_key]
    suite = benchmark.get_benchmark_dict()[spec['suite']](task_order_index=0)
    task_id, task = find_task(suite, spec['name'])
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    actual = file_hash(bddl)
    if task_key in BDDL_HASHES and actual != BDDL_HASHES[task_key]:
        raise ValueError(f'BDDL mismatch for {task_key}: {actual}')
    horizon = spec['horizon'] if protocol == 'reference' else {'libero_spatial':220, 'libero_object':280, 'libero_goal':300, 'libero_10':520}[spec['suite']]
    kwargs = dict(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    if protocol == 'reference':
        kwargs.update(hard_reset=False, horizon=horizon+11)
    env = OffScreenRenderEnv(**kwargs)
    if protocol == 'reference':
        # Match the reference experiment's renderer: disable multisampling.
        sim = env.env.sim
        sim.model.vis.quality.offsamples = 0
        context = sim._render_context_offscreen
        context.con.free()
        context._set_mujoco_context_and_buffers()
    low, high = env.env.action_spec
    import numpy as np
    if np.shape(low) != (7,) or not np.all(low == -1) or not np.all(high == 1):
        env.close()
        raise ValueError('Unexpected LIBERO controller action contract')
    return env, task, task_id, suite, horizon


def reset_episode(env, seed, initial_state=None):
    import numpy as np
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    sampled = np.asarray(env.get_sim_state()).copy()
    for _ in range(10):
        obs, _, _, _ = env.step([0.]*6 + [-1.])
    settled = np.asarray(env.get_sim_state()).copy()
    return obs, {'sampled_reset_hash':hashlib.sha256(sampled.tobytes()).hexdigest(),
                 'reset_hash':hashlib.sha256(settled.tobytes()).hexdigest()}


def video_frame(obs):
    import numpy as np
    from turbovla.evaluation.policy import rotate_libero_image
    return np.concatenate([rotate_libero_image(obs['agentview_image']),
                           rotate_libero_image(obs['robot0_eye_in_hand_image'])], axis=1)


def run_episode(env, policy, instruction, seed, horizon, *, initial_state=None, video_path=None, clip_actions=True):
    """Observe → infer a chunk → execute 12 actions → re-observe; success from simulator."""
    import numpy as np
    import imageio.v2 as imageio
    from turbovla.evaluation.policy import set_seed_everywhere
    if seed is not None:
        set_seed_everywhere(seed)
    obs, reset_info = reset_episode(env, seed, initial_state)
    actions = deque()
    writer = None
    if video_path is not None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=20, codec='libx264', quality=7, macro_block_size=16)
    action_log, latency, clipped = [], [], 0
    success = False
    try:
        if writer:
            writer.append_data(video_frame(obs))
        for step in range(horizon):
            if not actions:
                start = time.perf_counter()
                chunk = policy.predict_env_action_chunk_from_obs(obs, instruction, execute_steps=12)
                latency.append(time.perf_counter()-start)
                if chunk.shape != (12,7) or not np.isfinite(chunk).all():
                    raise ValueError(f'Expected finite (12,7) action chunk: {chunk.shape}')
                if clip_actions:
                    clipped += int(np.count_nonzero(np.abs(chunk)>1))
                    chunk = np.clip(chunk,-1,1)
                actions.extend(chunk)
            action = actions.popleft()
            obs, _, _, _ = env.step(action.tolist())
            action_log.append(action.tolist())
            if writer:
                writer.append_data(video_frame(obs))
            success = bool(env.check_success())
            if success:
                break
    finally:
        if writer:
            writer.close()
    return dict(seed=seed, success=success, steps=len(action_log), horizon=horizon,
                truncated=not success, prompt=instruction, **reset_info,
                policy_calls=len(latency), policy_wall_ms=float(1000*np.mean(latency)),
                clipped_components=clipped, actions=action_log,
                video=str(video_path.relative_to(ROOT)) if video_path and video_path.is_relative_to(ROOT) else str(video_path) if video_path else None,
                video_frames=len(action_log)+1 if video_path else 0)


def summarize(rows):
    if not rows or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique episode seeds required')
    n = len(rows)
    successes = sum(bool(r['success']) for r in rows)
    p, z = successes/n, 1.959963984540054
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return dict(episodes=n, successes=successes, success_rate=p,
                ci95=[0.0 if successes == 0 else max(0,center-half),1.0 if successes == n else min(1,center+half)],
                mean_steps=sum(r['steps'] for r in rows)/n)


def validate_run(task_keys, episodes, protocol):
    if protocol not in ('reference','official') or episodes < 1:
        raise ValueError('Invalid protocol or episode count')
    if protocol == 'official' and episodes > 50:
        raise ValueError('Official protocol has 50 benchmark initial states')
    if not task_keys or len(set(task_keys)) != len(task_keys) or any(k not in TASKS for k in task_keys):
        raise ValueError('Tasks must be a nonempty unique registered list')


def evaluate(policy, output, task_keys, episodes=50, seed_start=10000, protocol='reference', save_videos=True):
    validate_run(task_keys, episodes, protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve previous runs: choose a new output directory: {output}')
    output.mkdir(parents=True)
    write_json(output/'config.json',dict(tasks=task_keys, episodes=episodes, seed_start=seed_start if protocol=="reference" else None,
        rng_seed_once_per_task=7 if protocol=="official" else None,
        episode_key_meaning="RNG seed" if protocol=="reference" else "initial state index; actual RNG seed is rng_seed=7",
        protocol=protocol, precision=policy.precision, chunk_size=12, open_loop_steps=12,
        settled_steps=10, image_size=256, all_selected_tasks_seen_by_release=True if all(TASKS[k]['suite'] in ('libero_spatial','libero_object','libero_goal','libero_10') for k in task_keys) else None,
        clipped_actions=protocol=='reference', checkpoint_sha256=file_hash(ASSETS/'turbovla_libero.pth')))
    summaries = {}
    for key in task_keys:
        if protocol == 'official':
            from turbovla.evaluation.policy import set_seed_everywhere
            set_seed_everywhere(7)  # upstream seeds once per task, not each trial
        env, task, task_id, suite, horizon = make_env(key, protocol)
        if protocol == 'official':
            env.seed(7)
        rows=[]
        initial_states = suite.get_task_init_states(task_id) if protocol=='official' else None
        try:
            for i in range(episodes):
                seed = seed_start+i if protocol=='reference' else None
                # Record unique episode keys; official RNG remains fixed at seed 7.
                video = output/'videos'/key/f'episode_{i:03d}.mp4' if save_videos else None
                row = run_episode(env,policy,task.language,seed,horizon,
                    initial_state=initial_states[i] if initial_states is not None else None,
                    video_path=video,clip_actions=protocol=='reference')
                row.update(task=key,task_id=task_id,episode=i,rng_seed=seed if protocol=='reference' else 7,seed=seed if protocol=='reference' else i,
                           init_state_index=i if protocol=='official' else None)
                rows.append(row)
                write_json(output/key/'episodes.json',rows)
                if video:
                    write_json(video.with_suffix('.json'),{k:v for k,v in row.items() if k!='actions'})
                print(f'{protocol} {key} {i+1}/{episodes} success={row["success"]} steps={row["steps"]}',flush=True)
        finally:
            env.close()
        summaries[key]=summarize(rows)
        write_json(output/key/'summary.json',summaries[key])
        write_json(output/'summary.json',summaries)
    write_json(output/'complete.json',dict(tasks=task_keys,total_episodes=episodes*len(task_keys)))
    return summaries


def gpu_used_mib():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    return float(output.strip().splitlines()[0])


def memory_summary(samples, idle_mib):
    peak = max(s['used_mib'] for s in samples)
    delta = peak-idle_mib
    return dict(idle_mib=idle_mib,peak_mib=peak,delta_mib=delta,delta_gib=delta/1024,
                delta_gb=delta*1024**2/1e9,at_most_1_gib=delta<=1024,at_most_1_gb=delta*1024**2<=1e9)


def validate_idle_memory(used_mib, threshold_mib=32):
    """Headless-instance guard: container GPU PID reporting can be empty even under load."""
    if used_mib > threshold_mib:
        raise RuntimeError(f'GPU is not idle: {used_mib} MiB used (> {threshold_mib} MiB). Stop GPU kernels/evaluations before benchmarking.')
