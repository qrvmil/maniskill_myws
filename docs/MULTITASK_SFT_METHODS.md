# Multi-task SFT methods

This document specifies the primary experiment. Runtime evidence is collected in
`docs/multitask_sft/evidence`; large datasets and checkpoints live under
`/workspace/multitask-sft` on the execution instance. The experiment is not complete
until the main report contains measured results and the completion checks pass.

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
optimizer remain unchanged. Both Gemma LoRA variants use rank32/alpha32. Batch8,
seed0, no EMA, action horizon50, 3,001 updates. The inherited AdamW recipe has
b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, gradient clipping1.0. The inherited
cosine schedule uses 1,000 warmup steps, peak LR2.5e-5, decay horizon30,000 and
terminal LR2.5e-6. The schedule is not shortened to the experiment's update budget.
Exact config repr and runtime source snapshots accompany each run.

Save wrappers check the actual optimizer counter and preserve checkpoints at
0/500/1000/2000/3001 completed updates. No N−1 directory interpretation is used.
Update0 is the actual LoRA train state. Both LoRA factors are normally initialized
in this pinned implementation, so it contains a small random functional
perturbation relative to dense official base weights; it is not labeled dense
zero-shot performance.

## Normalization and deployment contract

Each variant fits statistics using only its own training corpus. The official
OpenPI chunk-weighted statistics path is used, including action-chunk overlap and
terminal padding. Statistics are computed over complete batches of8; this uses
5,832/11,744/16,776 frames for A/B/C respectively (the last single B frame is omitted
by the unchanged official statistics loader). These are mean/std normalization
statistics; no extra delta transform is applied to normalized OSC commands.
Existing statistics may be reused only when their recorded corpus and hashes match.

Initial statistics hashes:

| Variant | SHA256 |
|---|---|
| A | b4ea07567b2cc2bbd5670167efa540730ef4e98b3e507ce1610473c948401f44 |
| B | e5cbe62ae014e047c178279a53ce840efa40e1754c492dd1f492415feada29c3 |
| C | ecad97851b8c9330ddd6387791049fd512302aa75384c7ca36d2670e9cac6878 |

The action contract is LIBERO's seven normalized OSC controller commands, not
meters/radians. State is world EEF position, axis-angle orientation and two gripper
joint positions (8D), padded internally to32. Both cameras are resized/padded to224;
the third model camera is masked. Outputs are inverse-normalized exactly once,
trimmed to7 dimensions, clipped to[-1,1], and executed5 at a time before replanning.
Exact task text is checked immediately before the official tokenizer on every
inference call. The normalizer is loaded from the exact recorded file.

## Evaluation and behavior diagnostics

Every task uses generated-seed resets and10 settling actions. D0/D1/H1 use the
historical spatial horizon220; D2/H2 use the standard goal/object horizon280.
Task identity is enforced by registered BDDL hashes and exact simulator language.
The same task/seed reset hash is required across every checkpoint. A mismatched
reset fails the run; results are never silently paired across differing states.
Per-episode flow noise comes from the inherited independent NumPy generator.
JAX0.5.3 inference preserves `jax_cuda_autotune0_v1` with autotune level0.

D0/D1/H1 reuse the prior diagnostic definitions unchanged: ever EEF-to-target-body
3D distance <.10m; robosuite target-object grasp contact; target bowl rising >.03m
above post-settle height; and LIBERO binary success. Stages need not be nested.
Sampled contact can miss a physical grasp. D2/H2 are scored by task success without
misapplying bowl/plate-specific diagnostics.

## Statistics

Report successes/N, SR, pointwise Wilson95% intervals and mean episode length.
Comparisons join episode pairs by seed and exact reset hash. Rescue/harm counts
and mean gains are per task. Paired percentile bootstrap uses10,000 seed0 draws.
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

Use10 fixed initial observations each from D0 and H1 (seeds10000–10009). Cyclically
replace both cameras with those of the next seed in the same task. Keep the
receiver's proprioception, exact prompt and flow noise unchanged. Record bank and
noise hashes. Compare seven-dimensional inverse-normalized predicted actions
before controller clipping: first-action Euclidean distance, mean Euclidean
distance over the first5 actions, and mean over the full50-action chunk. This
measures behavioral image sensitivity, not attention, grounding correctness, or
whether larger perturbation improves task success.

## Videos

For each final model/task, retain the first2 successes and first2 failures in seed
order, or all available when a category has fewer. A zero category is recorded
explicitly. Videos show the two policy camera views at128px each and20fps, including
the terminal post-action observation (episode length+1 frames). Every video is
fully decoded and independently frame-counted with ffmpeg/ffprobe; row metadata,
seed, outcome, task, model, hashes and frame counts must match. Outcome-stratified
examples illustrate behavior; they do not determine any success-rate estimate.

## Runtime and optional control

Hardware: one NVIDIA CMP170HX,64GiB VRAM, driver610.43.03, system CUDA12.8.
Pinned environment: Torch2.7.1+cu128, JAX/JAXlib/CUDA plugin/PJRT0.5.3,
MuJoCo3.2.7, robosuite1.4.1, NumPy1.26.4. Package inventories and GPU usage sampled
every0.5s are recorded. Training runs serially under supervisor with JAX
preallocation disabled. No scientific hyperparameter changes are allowed in
response to memory or outcomes.

The preregistered optional secondary is the data-matched50-demo control: first25
numeric demonstrations per task for B, first17/17/16 for C; A is shared. It is run
only if affordable after the primary study. The main report must state whether it
was actually run. It is never mixed into the primary table. The primary comparison
confounds task diversity, total unique data, per-task update exposure and corpus
normalization; conclusions must preserve these distinctions.

The instance filesystem is not backed by a persistent volume. It survives a
stop/start but not recycle/destroy. Lightweight results are committed; checkpoints
remain explicitly located and hashed rather than being silently added to Git.
