# Multi-task SFT methods

This document specifies the primary experiment. Runtime evidence is collected in
`docs/multitask_sft/evidence`; large datasets and checkpoints live under
`/workspace/multitask-sft` on the execution instance. The experiment is not complete
until the main report contains measured results and the completion checks pass.

On 23 September 2026 the user requested immediate publication instead of waiting
for remaining intermediate evaluations. All 15 final N=50 cells and 60 image
perturbation measurements completed. Twelve of 36 intermediate cells completed:
all A/B/C update-0 D0/D1/D2 cells and A/update-500 D0/D1/D2. Interrupted B/update-500
saved 38, 39 and 29 episodes respectively. These partial rows remain in raw evidence
but are excluded from fixed-N summaries. Unstarted checkpoints remain missing.
The pipeline and its workers were stopped. The original preregistration is retained;
`multitask_sft/evidence/setup/user_requested_cutoff.json` records this user-directed deviation.
Publication uses `scripts/report_multitask_sft.py --allow-incomplete-trajectory`;
the option never permits incomplete final evaluations. Notebook cached mode uses
the same explicit option. Missing trajectory points are gaps, with no interpolation.

## Registration and source

The branch begins at `c9b14ef43f7a94213536ebb97f3056bfd9f9dd87` in
`qrvmil/maniskill_myws`. The supplied research request is preserved in
[multitask_sft/REQUEST.md](multitask_sft/REQUEST.md); the machine-readable registration
is [../configs/pld_libero/multitask_sft.json](../configs/pld_libero/multitask_sft.json).
OpenPI is pinned to `981483dca0fd9acba698fea00aa6e52d56a66c58`; LIBERO to
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. Prior audit results are unchanged.

Seeds 10000–10049 were registered before new evaluation results existed. An exact
search of seed fields in tracked configurations and prior-audit JSON found no
collision. Unavailable ignored outputs on other machines cannot be independently
checked. All final cells use N=50. Intermediate checkpoints evaluate D0/D1/D2 with
the same 50 seeds. H3 is not included. Final 3001-update checkpoints are selected
by design, without consulting evaluation outcomes.

## Training sets, sampling and data

A trains on D0; B on D0+D1; C on D0+D1+D2. Each task supplies all 50 native LIBERO
HDF5 demonstrations, so primary totals are 50, 100 and 150 demonstrations. Only
these three HDF5 files are downloaded. H1/H2 demonstrations are never read.
BDDL paths and hashes, exact prompts, numeric demo lists and lengths, source
checksums, and converted-file inventories are recorded separately.

Native observations follow the just-executed action. The inherited conversion
pairs `obs[i]` with `action[i+1]`, dropping the final observation. This yields
5,832 D0 frames, 5,913 D1 frames and 5,031 D2 frames. Cameras are rotated 180°.
The preflight checks both the first and last aligned frame of every demonstration
against its native camera arrays, state, float32 next-action label, and prompt.

A retains the historical shuffled single-task sampler. B and C draw a task
uniformly for each example, then draw from a shuffled cycling permutation of that
task's frames. Task probabilities are 1/2 or 1/3 regardless of trajectory length.
The dataset wrapper records the indices actually fetched. Optimizer-consumed
counts exclude the pinned trainer's final lookahead batch. Final frequencies must
be within 2 percentage points of the registered task probabilities; actual counts,
per-batch counts, and consumed indices are retained.

## Architecture, optimizer and budget

All three runs load the same verified official
`gs://openpi-assets/checkpoints/pi0_base` independently. Storage object sizes and
MD5 checksums are checked against the official inventory; local SHA256 manifests
are reverified before each training run. B never starts from A, and C never starts
from B. Every complete initialized parameter tree is hashed and compared with A,
including LoRA parameters.

The official pinned JAX trainer, model, freeze filter, image transformations and
optimizer remain unchanged. Both Gemma LoRA variants use rank 32 / alpha 32. Batch size 8,
seed 0, no EMA, action horizon 50, 3,001 updates. The inherited AdamW recipe has
b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, gradient clipping 1.0. The inherited
cosine schedule uses 1,000 warmup steps, peak LR 2.5e-5, decay horizon 30,000 and
terminal LR 2.5e-6. The schedule is not shortened to the experiment's update budget.
Exact config repr and runtime source snapshots accompany each run.

Save wrappers check the actual optimizer counter and preserve checkpoints at
0/500/1000/2000/3001 completed updates. No N−1 directory interpretation is used.
Update 0 is the actual LoRA train state. Both LoRA factors are normally initialized
in this pinned implementation, so it contains a small random functional
perturbation relative to dense official base weights; it is not labeled dense
zero-shot performance.

## Normalization and deployment contract

Each variant fits statistics using only its own training corpus. The official
OpenPI chunk-weighted statistics path is used, including action-chunk overlap and
terminal padding. Statistics are computed over complete batches of 8; this uses
5,832/11,744/16,776 frames for A/B/C respectively (the last single B frame is omitted
by the unchanged official statistics loader). These are mean/std normalization
statistics; no extra delta transform is applied to normalized OSC commands.
Existing statistics may be reused only when their recorded corpus and hashes match.

The [normalization comparison](multitask_sft/normalization_comparison.csv) records
the mean and standard deviation of every one of the eight state and seven action
coordinates for A/B/C. These 45 rows make the corpus-dependent scaling differences
inspectable alongside the statistics files and hashes.

Initial statistics hashes:

| Variant | SHA256 |
|---|---|
| A | b4ea07567b2cc2bbd5670167efa540730ef4e98b3e507ce1610473c948401f44 |
| B | e5cbe62ae014e047c178279a53ce840efa40e1754c492dd1f492415feada29c3 |
| C | ecad97851b8c9330ddd6387791049fd512302aa75384c7ca36d2670e9cac6878 |

The action contract is LIBERO's seven normalized OSC controller commands, not
meters/radians. State is world EEF position, axis-angle orientation and two gripper
joint positions (8D), padded internally to 32. Both cameras are resized/padded to 224;
the third model camera is masked. Outputs are inverse-normalized exactly once,
trimmed to 7 dimensions, clipped to [−1, 1], and executed 5 at a time before replanning.
Exact task text is checked immediately before the official tokenizer on every
inference call. The normalizer is loaded from the exact recorded file.

## Evaluation and behavior diagnostics

Every task uses generated-seed resets and 10 settling actions. D0/D1/H1 use the
historical spatial horizon 220; D2/H2 use the standard goal/object horizon 280.
Task identity is enforced by registered BDDL hashes and exact simulator language.
The same task/seed reset hash is required across every checkpoint. A mismatched
reset fails the run; results are never silently paired across differing states.
Per-episode flow noise comes from the inherited independent NumPy generator.
JAX 0.5.3 inference preserves `jax_cuda_autotune0_v1` with autotune level 0.

D0/D1/H1 reuse the prior diagnostic definitions unchanged: ever EEF-to-target-body
3D distance < 0.10 m; robosuite target-object grasp contact; target bowl rising > 0.03 m
above post-settle height; and LIBERO binary success. Stages need not be nested.
Sampled contact can miss a physical grasp. D2/H2 are scored by task success without
misapplying bowl/plate-specific diagnostics.

## Statistics

Report successes/N, SR, pointwise Wilson 95% intervals and mean episode length.
Comparisons join episode pairs by seed and exact reset hash. Rescue/harm counts
and mean gains are per task. Paired percentile bootstrap uses 10,000 seed 0 draws.
Constant-outcome bootstrap intervals are shown as degenerate, with an additional
exact boundary bound rather than an implication of zero uncertainty.

`HeldOutMean=(SR(H1)+SR(H2))/2` always gives equal task weights. Its bootstrap
resamples seeds within each fixed task, then averages task means; comparisons
preserve episode pairs within each task. Tasks are never pooled as independent
episodes. These intervals describe reset-sample uncertainty on the two specified
tasks, not uncertainty over an unseen-task population or over training seeds.
Seen/unseen averages list their task membership for each variant and never replace
the individual task results.

## Image-sensitivity diagnostic

Use 10 fixed initial observations each from D0 and H1 (seeds 10000–10009). Cyclically
replace both cameras with those of the next seed in the same task. Keep the
receiver's proprioception, exact prompt and flow noise unchanged. Record bank and
noise hashes. Compare seven-dimensional inverse-normalized predicted actions
before controller clipping: first-action Euclidean distance, mean Euclidean
distance over the first 5 actions, and mean over the full 50-action chunk. This
measures behavioral image sensitivity, not attention, grounding correctness, or
whether larger perturbation improves task success.

## Videos

For each final model/task, retain the first 2 successes and first 2 failures in seed
order, or all available when a category has fewer. A zero category is recorded
explicitly. Videos show the two policy camera views at 128 px each and 20 fps, including
the terminal post-action observation (episode length + 1 frames). Every video is
fully decoded and independently frame-counted with ffmpeg/ffprobe; row metadata,
seed, outcome, task, model, hashes and frame counts must match. Outcome-stratified
examples illustrate behavior; they do not determine any success-rate estimate.

## Runtime and optional control

Hardware: one NVIDIA CMP170HX, 64 GiB VRAM, driver 610.43.03, system CUDA 12.8.
Pinned environment: Torch 2.7.1+cu128, JAX/JAXlib/CUDA plugin/PJRT 0.5.3,
MuJoCo 3.2.7, robosuite 1.4.1, NumPy 1.26.4. Package inventories and GPU usage sampled
every 0.5 s are recorded. Training runs serially under supervisor with JAX
preallocation disabled. No scientific hyperparameter changes are allowed in
response to memory or outcomes.

Completed primary training runs (wall time includes initialization and saves):

| Model | Completed updates | Consumed D0 / D1 / D2 examples | Wall time | Sampled device peak |
|---|---:|---:|---:|---:|
| A | 3,001 | 24,008 / 0 / 0 | 8,609.94 s | 33,120 MiB |
| B | 3,001 | 12,020 / 11,988 / 0 | 8,558.16 s | 33,120 MiB |
| C | 3,001 | 7,954 / 8,080 / 7,974 | 9,357.18 s | 33,120 MiB |

Each run consumed 3,001 batches of 8; its final 8 fetched lookahead examples were
excluded. Saved sample indices independently reproduce the counts and per-batch
task frequencies. All five checkpoint manifests and their actual normalization
files were rehashed. Complete initialized parameter hashes match across A/B/C.

A container restart interrupted the first C attempt after its last logged step
1,480; its latest durable checkpoint was at 1,000 completed updates. The interrupted
checkpoints, runtime records and logs were preserved separately. C was restarted
independently from the official base with the original seed and recipe; the
interrupted attempt is excluded from the primary comparison. Recovery verified all
265 recorded package versions and actual Torch/JAX GPU operations before training.
One-time setup and experiment services no longer autostart concurrently after a
container restart; the pipeline is started explicitly after environment checks.
The incident and verification records are under
`multitask_sft/evidence/setup/recovery_20260922T200705Z`.

The preregistered optional secondary was the data-matched 50-demo control: the first
25 numeric demonstrations per task for B, and 17/17/16 for C; A would be shared.
It was not run. The completed primary B/C training runs took a combined 4.98 hours;
repeating those optimizer budgets for the control would add approximately 5 hours
of training before 500 additional final evaluation episodes. That would materially
extend the study. The design was selected before outcomes; the runtime-based
decision not to launch it was made after the primary final evaluations completed.
The primary comparison therefore leaves task diversity, total unique data,
per-task update exposure and corpus normalization confounded.

The instance filesystem is not backed by a persistent volume. It survives a
stop/start but not recycle/destroy. Lightweight results are committed; checkpoints
remain explicitly located and hashed rather than being silently added to Git.

## Parallel evaluation gate

Before using parallel evaluation, the launcher compares serial A/update 0/D0
rollouts on seeds 10000–10001 with three concurrent independent processes. Success,
episode length, reset hash, complete physics/action trajectory hash and camera
image hash must all match exactly for every process. This gate uses no H1/H2
outcomes and changes no model choice. A failure selects serial execution. A pass
permits at most three processes on distinct tasks, each with its own simulator,
policy and explicit per-episode noise. Final variants remain sequential. The
runtime records the gate's decision and source hashes; actual execution mode must
be read from that evidence rather than assumed from this design.

The gate passed on this instance: all three concurrent workers matched the serial
reference exactly for both registered seeds, including trajectory and camera
hashes. The saved rollouts were independently compared again. Primary evaluation
therefore uses up to three distinct tasks concurrently; this bounded check is
evidence for that execution mode, not an exhaustive numerical equivalence proof.

## Reusing the evaluator and notebook

From the repository root, the pinned environment is created by
`bash scripts/pld/setup_libero.sh`. On this instance, it is already installed at
`third_party/openpi/.venv`. The runtime verifies the JAX numerical contract before
constructing a policy. The notebook additionally uses `nbformat`, `nbclient`,
`nbconvert` and `ipykernel`, installed into that same environment; its kernel is
named `pi0-multitask`.

The following evaluates the final three-task model on H1 into a separate rerun
directory. Change the checkpoint, task, episode count, initial seed or video count
directly. `--seeds 10002,10005` supplies an explicit seed list instead.

```bash
export PYTHONPATH=/workspace/LIBERO:src
export HF_LEROBOT_HOME=/workspace/multitask-sft/lerobot
export MUJOCO_GL=egl JAX_PLATFORMS=cuda
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 TORCH_COMPILE_DISABLE=1
third_party/openpi/.venv/bin/python scripts/eval_multitask_sft.py \
  --checkpoint /workspace/multitask-sft/C/checkpoints/pi0_libero_seen_lora32/EXP-001/3001 \
  --variant C --task H1 --episodes 50 --seed-start 10000 --videos 2 \
  --output /workspace/multitask-sft/reruns/C-H1
```

Each output directory is single-use. A compatible external checkpoint can use
`--variant external` and its exact statistics file with
`--normalization /path/to/norm_stats.json`. With one statistics file under its
`assets` directory, discovery is automatic. A/B/C labels are inferred only from
their registered asset IDs; `--variant external` overrides that convention when
evaluating another checkpoint. Compatibility means the same π₀ LoRA32 architecture,
action horizon 50 and LIBERO state/action contract used here.

External mode records training exposure as unknown (`seen: null`) by default.
Supply `--seen-tasks D0,H1` to identify which of the five registered evaluation
tasks were included in that checkpoint's SFT, or `--seen-tasks none` when none were.
This does not alter the registered A/B/C training sets. External runs use serial
execution and default to `/workspace/multitask-sft/external_eval`, outside the
primary evidence tree. `--task all` evaluates all five tasks; primary runs can also
request serial execution with `--serial`.

The notebook's first code cell contains all editable evaluation settings. Its
default mode reads committed evidence on CPU. Set `LIVE_EVALUATION=True`, choose
the checkpoint and seeds, then restart the pinned kernel and run all cells for a
new GPU evaluation. Its final comparison tables remain the registered primary
experiment; exploratory rollouts have separate output directories.
For another compatible checkpoint, set `VARIANT="external"`, supply its
`CHECKPOINT`, and optionally set `NORMALIZATION` and `SEEN_TASKS` in that same cell.

The complete primary pipeline is `scripts/run_multitask_sft.py`, after fetching
the three sources with `scripts/prepare_multitask_sft.py fetch` and preparing each
variant with `scripts/prepare_multitask_sft.py prepare`. On this Vast instance it
runs through the recorded supervisor wrapper and configuration. Dataset and
checkpoint roots are defined in `multitask_data.py`; a fresh training reproduction
requires a fresh experiment root. The collector exports lightweight evidence, and
`scripts/report_multitask_sft.py` checks all required cells before writing the
statistics and six PNG/PDF figure pairs.
