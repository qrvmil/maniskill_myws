# pi0 D0/D1 sanity audit: methods and source-contract findings

This file records methods; the [human report](../PI0_D0_D1_SANITY_REPORT.md) explains the measured results.

## Models and transformations

The official storage API inventory for `pi0_base` contains normalization assets
for arx, arx_mobile, droid, fibocom_mobile, franka, trossen, trossen_mobile, ur5e,
and ur5e_dual, **not LIBERO**. The shared robot family does not justify reusing
Franka or DROID normalization: camera layouts, state coordinates, action units
and training transformations can differ. The bundled Franka/DROID statistics
have eight non-padding action coordinates; LIBERO outputs seven. The pinned
DROID adapter explicitly consumes seven joint positions plus one gripper value
and returns eight action coordinates, unlike LIBERO's Cartesian/axis-angle
state and seven OSC commands. Matching robot family or padded 32D model shape
does not establish an equivalent action contract. Raw pi0_base is therefore **not directly
evaluable under a valid LIBERO action contract** in this audit. This is an
absence of a justified deployment contract, not a measured zero success rate.

The primary pre-SFT baseline supplies the frozen D0 normalization and LIBERO
transforms to official dense pi0 weights. D0-specific statistics constitute
embodiment/data alignment despite zero gradient updates; this does not establish
a generic pi0 LIBERO zero-shot score. The later models use the same D0 statistics.

The pinned LoRA code initializes both factors with normal(stddev=.01). The
zero-update LoRA training state therefore contains a random functional
perturbation, and is evaluated separately. Its initialization and all training
settings remain exactly the V4 configuration. Comparing it with the dense primary
baseline separates initialization effects from gradient-driven changes.

Official positive control: `gs://openpi-assets/checkpoints/pi05_libero`, configured
by pinned `pi05_libero`, with `assets/physical-intelligence/libero/norm_stats.json`
loaded by `create_trained_policy` from that checkpoint. It is not admitted into
D0-only SFT. The [official LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
identifies this checkpoint and configuration. Native action horizon is 10 for
pi0.5 and 50 for D0 pi0; both execute the first 5 commands before replanning.

All models use the historical `LiberoEnv`, `openpi_observation`,
`AlignedOpenPIModel.infer`, `ChunkedBasePolicy`, and `run_episode`. Raw agentview
and wrist cameras are rotated 180°, resized with padding to 224 pixels, and supplied as
the two valid cameras with a masked third camera. State is 8D: world EEF position,
axis-angle orientation, and two gripper joint positions. The models internally
pad state/actions to 32 dimensions and return the first 7 action dimensions.

D0 and official control both disable `extra_delta_transform`; OSC commands already
represent normalized controller deltas. OpenPI uses mean/std normalization for D0 pi0 and 1st/99th-percentile
normalization for the official pi0.5 control, as prescribed by their respective
configs. Each performs the matching inverse normalization exactly once. Afterwards the
existing chunk adapter clips commands to [-1, 1]. These controller commands are
not displacements in meters or radians. The controller applies its own scales.

At each real inference call, the audit checks the exact language string immediately
before the official tokenizer and records the resulting token hash. The prompts are:

- D0: `pick up the black bowl from table center and place it on the plate`
- D1: `pick up the black bowl next to the plate and place it on the plate`

The BDDL task pair also relocates the distractor: D0 places target bowl1 at the
center and bowl2 next to the plate; D1 places target bowl1 next to the plate and
bowl2 next to the ramekin. The center is empty in D1. This is therefore not a
pure prompt intervention or a one-object-only scene change. Both goal predicates
refer to bowl1 on plate1. The task definitions are unchanged.

## Training and checkpoint indices

All 50 native D0 HDF5 demonstrations yield 5,832 pairs after the existing
`obs[i]`→`action[i+1]` shift. D0-only conversion and official chunked normalization
precede any D1 diagnostic read. Dataset and statistics hashes are verified.
Seed 0, LoRA rank 32 / alpha 32, batch 8, model/transforms, optimizer, schedule, total 3,001
updates, no EMA and no D1 training input are inherited from V4.

A save-only wrapper around the pinned JAX trainer captures the initial state and
updates 500 / 1,000 / 2,000 / 3,001. It verifies actual `state.step == loop_index+1` and names
new directories by completed updates. Historical V4 directory 3000 meant 3,001
updates and is untouched. The new save callback/retention policy changes storage,
not the optimizer or batches.

## Evaluation and uncertainty

Each model uses generated reset seeds 9000–9049 with the historical 10 settling
steps and 220-action horizon, N = 50 per task. Seeds are paired across checkpoints;
the report generator rejects mismatched initial-state hashes. This seed block
was absent from historical tracked registrations, but historical ignored raw
outputs were not present on this new instance for independent inspection.

Success uses LIBERO's binary task check. Intervals are pointwise Wilson 95%.
Paired gain summaries use 10,000 seed-0 percentile-bootstrap resamples of episode
pairs; zero-discordance cases use the exact 95% discordance upper bound instead
of presenting a degenerate bootstrap interval. All-gain/all-loss boundary cases
use an exact binomial mass bound with worst-case opposite discordance. A baseline
compared with itself is an identity, with exactly zero difference. Repeated checkpoints are not
independent replications; these intervals do not measure training-seed variance.

Reach is ever 3D EEF-to-target-body distance < 0.10 m. Grasp is robosuite's target-object
contact check. Lift means the target body rises > 0.03 m over its post-settle height.
Placement is binary task success. Stages are descriptive and not necessarily
nested. The contact flag is sampled at action boundaries and can miss a physical
grasp: the official control registered 49 D0 grasps despite 50 successes, and 45
D1 grasps despite 47 successes. A false flag is not proof that no physical grasp
occurred. Target object is `akita_black_bowl_1`; destination is `plate_1`.

For seeds 9000–9009, compare the first complete predicted chunk and first 5 executed
commands. Mean L1 is the sum of absolute differences over 7 action dimensions,
averaged across states and time; mean L2 is the Euclidean command-vector difference.
The complete predicted chunk is before controller clipping; executed actions are
after clipping. Identical noise and initial scenes are required.

The user's later video requirement is recorded in the
[video addendum](../plans/2026-09-16-pi0-video-addendum.md). Every model/task retains
the first two successes and first two failures in seed order, or all available
episodes when a category has fewer than two. These outcome-stratified examples
are illustrative; the full 50-episode sample determines every reported rate.
Clips show agentview and wrist views side by side at 128 pixels per view and
20 frames/second. Each includes the terminal post-action observation, giving
episode length plus one frames. Completed baseline/control episodes are replayed
only for video capture, with exact seed/outcome/length and reset/trajectory/image
hash checks; these replays neither replace nor augment original result rows.
Publication copies are fully decoded, frame-counted and checksum-verified, and
the video index requires a visible-behavior note for each clip.

The read-only distribution audit compares temporally aligned unique native frames
from 50 D0 and 50 D1 demos. It reports D1 excursions beyond D0 observed min/max,
D0 1st/99th percentiles, and |D0-model z score|>3. Official normalization uses
chunk overlap/padding weighting, which differs from unique-frame weighting.
D1 native demos are distribution references; their replay equivalence is not
claimed. No statistic is adapted from this diagnostic.

## Runtime provenance

Actual hardware: one NVIDIA CMP 170HX, 64 GiB, driver 610.57.04, CUDA toolkit 12.8;
Torch 2.7.1+cu128 and JAX/JAXlib/CUDA plugin/PJRT 0.5.3; MuJoCo 3.2.7,
robosuite 1.4.1. LIBERO revision 8f1084e3132a39270c3a13ebe37270a43ece2a01;
OpenPI revision 981483dca0fd9acba698fea00aa6e52d56a66c58.

Inference preserves `jax_cuda_autotune0_v1` and uses explicit per-episode
NumPy sampling noise. GPU stages run serially under supervisor. JAX preallocation
is disabled; whole-device memory is sampled every 0.5 s by the historical logger.
A 64 GiB GPU cannot alone demonstrate execution on 48 GiB hardware; sampled peak
training usage was 32.34 GiB. No OOM occurred and no scientific settings were reduced.

Storage inventories and checksums verify 49 official checkpoint files. Exact
source/config snapshots are captured for subsequent stages. The positive-control
run began before the audit-entrypoint snapshot fix; historical modules were
snapshotted at launch and the later entrypoint capture is labeled accordingly.
Each learned checkpoint evaluation also records parameter-file SHA256 values
before loading the model, and binds that manifest into its evaluation metadata.

The original training metadata records the old, unused CLI default
`model: positive_control`. That flag was only consumed by evaluation; it did not
select the training model. The recorded training configuration identifies
`pi0_libero_seen_lora32`, `pi05=False`, the `pi0_base/params` weight loader, and
`local/pi0_audit_d0` as the sole dataset. Original metadata is preserved. The CLI
now reserves `--model` for evaluation and uses no model default for training.

Lightweight contract tests overlapped the positive-control run; its sampled
whole-device peak may include those test processes. Long model evaluation and
SFT stages run serially, and training has no concurrent evaluator.
