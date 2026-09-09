# EXPERIMENT_DESIGN — PLD LIBERO residual transfer

Scientific split frozen before policy results, 2026-09-09. Hardware scope amended by explicit user instruction: use the actual A100 80 GB; previous 16 GiB probes remain historical engineering results. Machine-readable split:
`configs/pld_libero/anchor_bowl.json`; task BDDL snapshots and hashes in EXP-000/audit.

## Research question
How far across task space does a source-task residual correction help a frozen aligned pi0 before gain disappears or turns negative?

## Hypothesis
H1: residual RL improves source-task success. H2: gain decreases with task distance.
H3: sufficiently distant tasks have approximately zero gain. H4: negative transfer may occur.
These are hypotheses, not results.

## Method
pi0_pretrained → SFT(T_seen) → frozen pi_b → residual RL(T_seen) → paired zero-shot evaluation(T_unseen).
**NO DISTILLATION.** One task-specific visual residual specialist per source. No new language-conditioned residual. The base receives target instructions at evaluation; the residual sees images, 8D proprioception and 7D base action.

## Data split
T_seen is only the D0 task below. Every other listed task ID is evaluation-only.
Training environment seeds 1000–1099, source-validation seeds 2000–2019, final evaluation seeds 3000–3049. Training cycles the registered training seed set; no hidden seed expansion.
SFT: only source demonstrations. Evaluation uses fresh generated resets with disjoint seeds, NOT benchmark init files potentially associated with demonstrations. Source D0 is same-task generalization, never unseen-task generalization. Reset-state fingerprints must be saved; exact duplicate initial-state detection against SFT states is required before final D0 claims.

## Task-distance definition
Ordinal, interpretable buckets, not a metric or a causal isolation of semantic distance. Layout and horizon are confounders. Horizon column is the official OpenPI suite evaluation cap, not a measured episode length. Subgoals are required skills; they need not equal BDDL predicate count. D5 is strongly compositional and has a larger budget; it is not proven intrinsically farther than every D3 task.

| Distance | Suite | Task ID | Exact task name | Object | Destination | Relation | Primitive | Skill subgoals | Step cap | Language/layout changes and reason |
|---|---|---:|---|---|---|---|---|---|---:|---|
| D0 | libero_spatial | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | black bowl | plate | center → on plate | pick/place | 1 | 220 | anchor; two bowls, spatial grounding |
| D1 | libero_spatial | 8 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | black bowl | plate | next to plate → on plate | pick/place | 1 | 220 | same primitive/object/destination; source location and distractor placement vary |
| D1 | libero_spatial | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | black bowl | plate | next to ramekin → on plate | pick/place | 1 | 220 | same primitive/object/destination; spatial phrase and bowl placement vary |
| D2 | libero_goal | 1 | put_the_bowl_on_the_stove | black bowl | stove | on stove | pick/place | 1 | 300 | destination semantics and layout change |
| D2 | libero_object | 0 | pick_up_the_alphabet_soup_and_place_it_in_the_basket | alphabet soup | basket | in basket | pick/place | 1 | 280 | object, destination, language and layout change |
| D3 | libero_goal | 3 | open_the_top_drawer_and_put_the_bowl_inside | bowl + drawer | top drawer | inside | open then pick/place | 2 | 300 | opening prerequisite; BDDL has one final In predicate, not two success predicates |
| D4 | libero_goal | 0 | open_the_middle_drawer_of_the_cabinet | middle drawer | cabinet | open | articulation | 1 | 300 | new primitive and goal language; shared tabletop assets but changed layout |
| D4 | libero_goal | 5 | push_the_plate_to_the_front_of_the_stove | plate | stove front | front of | push | 1 | 300 | contact manipulation without bowl grasp; goal/layout change |
| D5 | libero_10 | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | bowl + drawer | bottom drawer | inside + closed | pick/place then close | 2 | 520 | kitchen scene; initially open drawer; In AND Close goals, longer evaluation budget |

## Leakage rules
Unseen definitions may be read to specify distance and construct evaluation environments. Unseen demonstrations, normalization statistics, rewards, rollouts and success rates must not influence SFT, offline replay, RL, hyperparameters or checkpoint selection. No full-LIBERO aligned checkpoint in primary runs. Training/collection must reject a non-source task. Manifest records source ID, demo hashes, statistics hashes, pretrained checkpoint and aligned checkpoint. A missing alignment manifest blocks scientific rollout commands. Unseen evaluation begins only after fixing the checkpoint. Additional anchors require separate specialists and preregistered ladders.

## Base-policy alignment
Preferred pretrained checkpoint: `gs://openpi-assets/checkpoints/pi0_base` (not pi0/pi05_libero). Official OpenPI LIBERO LeRobot format; source-only physical subset with source-only normalization.
Raw action: 7D normalized robosuite OSC_POSE input (delta xyz, delta axis-angle, gripper), [-1,1]. Controller maps to physical deltas; do not apply joint-position conventions from ManiSkill. State: eef xyz + quaternion converted to axis-angle + two gripper qpos = 8D. Cameras ordered agentview, robot0_eye_in_hand; render 256, rotate both axes 180 degrees per official evaluation, resize/pad to 224. Model adds masked third camera. Language comes from benchmark task.language.
Use `extra_delta_transform=False` for newly trained checkpoint because LIBERO actions already are deltas. This differs from upstream legacy pi0_libero config with extra delta enabled; training and inference must agree. OpenPI owns standardization/inverse-standardization; residual operates in bounded environment units AFTER inverse transform.
Current A100 SFT: all 3,501,372,176 pi0 parameters remain trainable; pinned official OpenPI PyTorch train_loop, selective bf16, batch 1, no EMA, gradient checkpointing, GPU forward/backward and GPU AdamW. Earlier GPU AdamW and official JAX LoRA failed 16 GiB-constrained probes; CPU AdamW passed, but the user subsequently authorized using the actual A100 80 GB. Corrected native corpus v3 has 5,832 observation/action pairs from exactly 50 source demonstrations (5,882 raw frames). Official source-only normalization is bound to that corpus. LR uses official warmup 1,000 steps toward 2.5e-5, with the official cosine schedule thereafter; ordinary AdamW and gradient clipping 1. No gradient accumulation. First source-validation pilot: 100 total steps; planned extended budget: 3,000, save every 500/final. A two-step checkpoint is an engineering fixture, not evidence of adequate alignment. A100 measurements do not establish Blackwell fit or throughput.

## Residual RL
Reuse `pld/sac.py` and CPU `pld/replay_buffer.py`. Actor: existing Gaussian tanh residual, three 256-wide LayerNorm/Tanh MLP layers. Visual path: existing ResNetV1-10-like GroupNorm encoder, cameras concatenated by channels; 256D visual latent, 128px uint8 CPU replay. Separate actor/twin Q/target encoder copies are existing architecture; only one frozen VLA.
a_exec = clip(a_base + 0.5*tanh(u), -1, 1). Bound is per dimension before and after clipping; no double scale. Critic sees executed action, not a mislabeled residual action.
Offline data: successful actual frozen aligned-base trajectories on source only. Log attempts, successes, transitions and MC returns r_t + gamma*(1-done)*R_(t+1); sparse terminal success reward 1, otherwise 0. Store actual base and next-base actions; never use legacy ManiSkill H5 or human actions as a base proxy. Timeout ends finite-horizon episode (done for backup); document this convention.
Cal-QL: existing critic-only warmup, gamma .99, alpha 5, 10 candidates, 1000 exploratory updates. Actor unchanged in warmup. Online: residual SAC 50000 steps, batch 8, CPU capacity 10000, offline:online 1:1, one update/step, actor every 2 critic updates, learning rates 3e-4, tau .005. First smoke: batch 2, 2 candidates, 2 Cal-QL and 2 SAC updates; no scientific training claim.
A: no OTF, first debug/baseline. B: 1 sampled residual edit + exact unedited base, argmax min(Q1,Q2); target backup uses max min target-Q without entropy. Base action held fixed; does not resample base VLA candidates. Evaluate deterministic actor for primary transfer, explicitly distinct from OTF rollout policy; seeded stochastic OTF may be a separate diagnostic, never mislabeled deterministic actor.

## Evaluation
Paired same environment seed and same reset state for base/residual; separate model RNG reset per episode, isolated from residual RNG. Replan every 5 actions; cache advanced exactly once per step, cleared on reset. Deterministic residual actor. Save episode records and reset fingerprints, base/residual successes, lengths, action bounds/clipping, failures. Zero-residual full-trajectory equivalence is required first. 2 episode smoke; 50 paired episodes/task for exploratory report. Choose final checkpoint before unseen runs.

## Metrics
Primary G(T)=SR_residual(T)-SR_base(T), report in fractions and percentage points. G(d) is equal task-weight mean within each source/distance bucket. Never hide negative values. Secondary: absolute SR, episode length, return, runtime, memory. No rates for unrun experiments.

## Statistical protocol
Training seed 0 initially: exploratory single-seed, not significant evidence. Final desired seeds 0,1,2 using the same frozen split and paired evaluation seeds. Keep per-anchor and per-seed results before aggregation. Paired bootstrap intervals across episode differences for exploratory uncertainty; multiple training seeds required for broader conclusions. A one-episode gain is 2 percentage points at n=50.

## Hardware constraints
Current authorized hardware: **one NVIDIA A100 80 GB PCIe**, driver 595.71.05, experiment environment torch 2.7.1+cu128, CUDA 12.8 (system venv torch 2.11.0+cu128). User explicitly changed the hardware scope on 2026-09-09. Current runs may use the actual 80 GB; prior 16 GiB-cap results remain feasibility diagnostics only and are not RTX 5080 validation. One environment and one base model initially, CPU uint8 replay, sequential rollout/update. Measure inference/update/combined allocation, reserved and sampled device memory, wall times and CPU RAM. Workspace is not volume-backed.

## Deviations from PLD paper
| Paper | Our implementation | Reason | Expected consequence |
|---|---|---|---|
| Three-stage PLD including distillation | Stop after specialist acquisition; directly test frozen residual | Research question | Does not reproduce published distilled generalization numbers |
| Many source specialists/data collection scale | One anchor, one environment, one seed initially | Controlled initial experiment; originally 16 GB target | Higher variance and lower throughput |
| Pretrained ResNetV1-10 visual residual | Existing randomly initialized channel-stacked GroupNorm ResNet-like encoder | Reuse unofficial reproduction; no matching pretrained weights bundled | Material representation/pretraining mismatch, potentially weaker transfer |
| Successful frozen-base offline rollouts | Same requirement; new LIBERO buffer, no legacy proxy | Avoid fidelity regression | Can block if aligned base has no successes |
| Paper sparse reward and training hyperparameters | Sparse terminal reward (matches paper main runs); inherited SAC/Cal-QL settings | Explicit practical baseline | Optimization may differ substantially |
| OTF defaults to 1 residual sample around fixed base | 1 residual sample plus explicit unedited base candidate, min-Q argmax and hard-max target | Closest existing reproduction path | Fixed-base sampling matches paper; extra base candidate/entropy treatment remain reproduction choices |
| Paper-scale batches/update ratio | Batch 8, 1 update/step initially | Conservative initial configuration, retained after A100 authorization | Learning speed/stability may differ |
| Standard benchmark evaluation init files | Disjoint seeded fresh initializations | Stronger same-task holdout from demos | Not directly comparable to official benchmark SR |
| Paper appendix uses rank-32 LoRA on 8 L40s | Full pi0 SFT with official GPU PyTorch trainer on user-authorized A100; batch 1/checkpointing/no EMA | Requested full-model alignment; 16 GiB fallback retained separately | Alignment differs from paper LoRA; batch 1 may require more updates |
| LIBERO paper settings | Existing tanh MLP, AdamW, entropy scaling, separate visual encoders | Preserve reproduction | Not authors' exact code |

## Planned experiment table
| ID | Purpose | Gate |
|---|---|---|
| EXP-000 | audit, unit tests, environment/zero-residual/dummy smoke, memory probes | no learned claims |
| EXP-001 | source-only alignment and paired base/source evaluation; base offline collection | verified SFT provenance |
| EXP-002 | source residual A then B, Cal-QL + online visual SAC | successful aligned-base buffer and finite smoke |
| EXP-003 | D1 transfer | frozen source checkpoint |
| EXP-004 | D2 transfer | same checkpoint |
| EXP-005 | D3 transfer | same checkpoint |
| EXP-006 | D4 transfer | same checkpoint |
| EXP-007 | D5 transfer | same checkpoint |
| EXP-008 | seeds 1,2 / additional anchors | per-anchor pipeline passes |

## Repository audit and implementation sequence
1. Reuse SAC/Cal-QL/replay/state image helpers; current orchestration, environment-device setup, H5 schema and observation bridge are ManiSkill-specific.
2. Add isolated `pld/libero_backend.py` (env, observations, actions), `pld/libero_protocol.py` (frozen split/provenance), `pld/libero_runner.py` (paired evaluation/collection/training), and `scripts/pld/run_libero.py` CLI. Leave legacy commands working.
3. Test contracts first in `tests/test_pld_libero.py`; run real reset/step/zero equivalence before learned inference.
4. Use official source-filtered LeRobot conversion/normalization and OpenPI training config; record exact subset and settings, probe memory before any fallback.
5. Gate aligned-base collection, finite Cal-QL/SAC smoke, source training, then unseen paired evaluation. Preserve all failed runs and raw metadata.

## Sources
- PLD: https://arxiv.org/html/2511.00091v1 (paper audit continues before main training).
- LIBERO official task definitions: https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/bddl_files
- OpenPI: https://github.com/Physical-Intelligence/openpi ; repository-pinned fork at `third_party/openpi`, revision in EXP-000/audit.

## Data and implementation contracts
- Source HDF5: 50 demonstrations, 5882 raw frames, SHA256 `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`. Only this task file was downloaded from the official LIBERO mirror named by its download utility. HDF5 BDDL metadata must match the frozen source.
- Native creation stores observation[i] after action[i]. Converter version `native_post_action_shift_v1` pairs observation[i] with action[i+1], yielding 5832 pairs, and drops the terminal observation. Earlier same-index checkpoints are rejected. No no-op filtering; 20 Hz control and LeRobot FPS 20. This differs from the official modified RLDS example's no-op filtering/FPS 10.
- Demonstration cameras are 128x128; runtime renders 256; both are rotated 180 degrees and OpenPI resize/pads to 224. This native/runtime resolution difference and disabled runtime MSAA are explicit image-distribution deviations.
- Exact source-state holdout checks include each native first recorded state and first observed successor state, excluding simulation-clock values. Disjoint fresh seeds further separate training/validation/final evaluation.
- Converted dataset files, source audit, repo ID/root, normalizer producer/statistics, and aligned checkpoint files are checksum-bound. Action cadence, rendering, image size and residual scale are bound to zero checks, offline buffers and specialists. Resume additionally binds seed, optimizer configuration and pretrained weights.
- Pretrained PyTorch conversion is the official OpenPI converter. An unused expert lm_head has random initialization absent from the JAX checkpoint; functional tensors are verified separately and actual whole-output hashes retained.
- PyTorch inference is eager (`TORCH_COMPILE_DISABLE=1`) to avoid autotuning caches. Frozen base parameters explicitly have requires_grad=False. Transformers' no_init_weights skips discarded initialization while preserving CPU buffers; strict loading and explicit weight tying remain required.
- Soft physics resets reuse the renderer, but placements are freshly sampled from the registered seed. Offscreen MSAA is disabled after longer tests exposed RGB nondeterminism. Zero-equivalence threshold fixed before aligned inference: max action/full simulator-state absolute difference <=1e-5, equal reset hash/length/success, and RGB difference <=1 uint8 intensity. Completed zero tests retain actual differences.
- Runtime dependencies are pinned by the setup script and saved per run. CUDA 12.8 wheels support the target architecture; only A100 execution has been measured here.

## Runnable commands
Run from the repository root. `scripts/pld/setup_libero.sh` provisions the pinned simulator/OpenPI environment with CUDA 12.8 wheels. `scripts/pld/libero_python.sh` selects this environment, source paths, EGL, CPU JAX and eager PyTorch. Every output directory must be new; never overwrite prior runs. The commands below are templates for pending stages, not evidence they ran.

```sh
# New artifact names: these paths must not already exist.
PLD_SOURCE_H5=/workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5
PLD_DATA_REPO=local/pld_libero_bowl_fresh
PLD_DATA_ROOT=/workspace/datasets/lerobot/$PLD_DATA_REPO
PLD_BASE=/workspace/checkpoints/pi0_base_pytorch_fresh
PLD_ALIGN=outputs/pld_libero/EXP-001/sft-fresh
PLD_ZERO=outputs/pld_libero/EXP-001/zero-fresh
PLD_OFFLINE=outputs/pld_libero/EXP-001/offline-fresh
PLD_SPECIALIST=outputs/pld_libero/EXP-002/source-otf-fresh

scripts/pld/libero_python.sh scripts/pld/fetch_libero_source.py
scripts/pld/libero_python.sh scripts/pld/prepare_pi0_libero.py --output "$PLD_BASE"
scripts/pld/libero_python.sh scripts/pld/align_libero.py prepare --source-h5 "$PLD_SOURCE_H5" --repo-id "$PLD_DATA_REPO" --dataset-root "$PLD_DATA_ROOT" --output outputs/pld_libero/EXP-001/prepare-fresh
scripts/pld/libero_python.sh scripts/pld/align_libero.py norm --source-h5 "$PLD_SOURCE_H5" --repo-id "$PLD_DATA_REPO" --dataset-root "$PLD_DATA_ROOT" --method full_torch --output outputs/pld_libero/EXP-001/norm-fresh
scripts/pld/libero_python.sh scripts/pld/align_libero.py train --source-h5 "$PLD_SOURCE_H5" --repo-id "$PLD_DATA_REPO" --dataset-root "$PLD_DATA_ROOT" --method full_torch --steps 3000 --pytorch-base-checkpoint "$PLD_BASE" --output "$PLD_ALIGN"
scripts/pld/libero_python.sh scripts/pld/run_libero.py base --validation --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --episodes 10 --output outputs/pld_libero/EXP-001/validation-fresh
# If source validation yields no successes, extend seen-only alignment before continuing.
scripts/pld/libero_python.sh scripts/pld/run_libero.py zero --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --episodes 2 --output "$PLD_ZERO"
scripts/pld/libero_python.sh scripts/pld/run_libero.py collect --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --zero-report "$PLD_ZERO/eval/zero_equivalence.json" --successes 50 --max-attempts 100 --output "$PLD_OFFLINE"
scripts/pld/libero_python.sh scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_rl_smoke.json --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --zero-report "$PLD_ZERO/eval/zero_equivalence.json" --offline-buffer "$PLD_OFFLINE/offline.npz" --output outputs/pld_libero/EXP-002/rl-smoke-fresh
scripts/pld/libero_python.sh scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --zero-report "$PLD_ZERO/eval/zero_equivalence.json" --offline-buffer "$PLD_OFFLINE/offline.npz" --output "$PLD_SPECIALIST"
scripts/pld/libero_python.sh scripts/pld/run_libero.py eval --config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest "$PLD_ALIGN/alignment_manifest.json" --zero-report "$PLD_ZERO/eval/zero_equivalence.json" --checkpoint "$PLD_SPECIALIST/checkpoints/residual_step_50000.pt" --distance D1 --episodes 50 --output outputs/pld_libero/EXP-003/paired-fresh
```
Use D0 for source paired evaluation under EXP-002, and D2/D3/D4/D5 with EXP-004/005/006/007. The same frozen specialist is used throughout. Default config is A (no OTF); separate OTF config uses paper-default residual candidate count 1 and warmup 100 episodes; debug A defaults to 5 warmup episodes. The short scientific RL config still requires a real successful aligned-base buffer.

### Renderer amendment
Disabling offscreen MSAA (`offsamples=0`, default upstream 4) eliminated RGB mismatch in a 220-step diagnostic and the subsequent six-episode smoke. This alters anti-aliasing relative to native demonstration images, and is a documented observation-rendering deviation. Physics, controller and task definitions are unchanged. The conservative numerical zero gate remains mandatory for each aligned checkpoint.

### Final paper audit correction before residual training
[PLD Appendix B.2/C](https://arxiv.org/html/2511.00091v1#A2.SS2) explicitly samples residuals around a **fixed** base action and defaults to OTF=1; holding the base fixed is therefore not itself a deviation. The initial draft overstated that difference. Main B now uses one residual candidate plus the reproduction's explicit base candidate, and 100 base warmup episodes. A remains the smaller debugging baseline. The paper uses pretrained visual encoders; our inherited encoder starts randomly, a material deviation. Appendix C also describes LoRA SFT; this project's full-model alignment follows the user's requested fidelity preference after measured memory probes, not a claim that the paper required full SFT. These changes were registered before any residual training or unseen evaluation.

### Corrected alignment pilot and feasibility
Two corrected-data full-model updates completed under the 16 GiB allocator cap using CPU AdamW; exact measurements belong in RESULTS.md. The native timing bug invalidated earlier scientific checkpoints, whose runs remain in the append-only ledger. Restart from official pretrained pi0 using corpus v3.

The first corrected pilot is fixed at **100 total updates**, after a two-update checkpoint and zero-residual engineering gate. It is an intermediate source-competence check before extending toward3000, not a final aligned-base claim. The managed supervisor job must point to corpus v3 and a corrected checkpoint. CPU threads16, ordinary unfused AdamW; fused AdamW was not selected because a bf16 microbenchmark showed rounding differences. Storage path is selected only after the full-model equivalence comparison; it does not change trainability or equations.

Source-only alignment selection uses `run_libero.py base --validation --episodes 10`, drawing seeds 2000–2009. The default base/eval commands retain final seeds 3000–3049. D0 final seeds are not used for alignment selection.

### Residual implementation details retained/registered before training
- Main B interprets the paper's target entropy -7/2=-3.5 in **unit residual coordinates**, before multiplication by scale. The inherited actor includes scale in its density Jacobian, so its configured target is `-3.5 + 7*log(0.5) = -8.352030263919616`. The paper does not specify its density coordinate convention; this is an explicit interpretation, not verified equivalence to author code. The inherited `+1e-6` Jacobian stabilizer makes this conversion approximate near tanh saturation. Debug A retains the reproduction default `-7 + 7*log(0.5)` (approximately -11.852). Initial temperature1, gradient clip1 and actor interval2 are unchanged. Final executed-action clipping does not receive a further density correction.
- Existing Cal-QL uses uniform, current-residual and next-residual action candidates, evaluates the latter under the current state, applies MC-return lower bounds to policy candidates, importance-density correction and logsumexp conservatism. Its random residual actor is held unchanged during warmup. This is the inherited practical Cal-QL estimator, not a verified reproduction of the authors' SERL implementation.
- Offline Cal-QL max-Q backups and main OTF backups omit entropy. Online A uses the inherited entropy-regularized SAC target. OTF base candidate has log-probability placeholder0; main hard-Q selection does not use that placeholder.
- The fixed-cap finite-horizon terminal mask suppresses bootstrapping at timeouts. Base action chunk phase is cached in the wrapper but not separately observed by the residual critic; critic inputs remain the reproduction's visual/state/action inputs. This is an approximation when using cached VLA chunks.

### Completed-run aggregation
After learned residual evaluations finish, recompute metrics from paired episode records:
```sh
scripts/pld/libero_python.sh scripts/pld/summarize_libero.py --runs outputs/pld_libero/EXP-002/source-paired-001 outputs/pld_libero/EXP-003/paired-001 outputs/pld_libero/EXP-004/paired-001 outputs/pld_libero/EXP-005/paired-001 outputs/pld_libero/EXP-006/paired-001 outputs/pld_libero/EXP-007/paired-001 --output outputs/pld_libero/EXP-007/aggregate-001
```
The summarizer rejects failed/dummy runs, duplicate targets, and different checkpoints within a source/training-seed ladder. It checks signed gains against raw paired episodes and emits task/bucket CSVs plus JSON listing missing tasks. Bucket gains remain per source and training seed; no single-seed significance claim. New run artifacts include local implementation snapshots, dependency versions and a scoped git diff as well as revision/dirty status.

### Optional optimizer storage engineering test
`align_libero.py --optimizer-storage resident_cpu` retains one model on GPU and stores AdamW parameters/states on CPU, copying gradients down and updated weights back. It changes storage/movement, not trainability, optimizer equations, dtypes or data. A three-update small-model CPU comparison matches ordinary moving-model AdamW exactly; the corrected two-update full-model GPU comparison matched all 777 saved tensors exactly. Resident CPU optimizer storage is selected for the corrected 100-update pilot: two measured update times 12.99/12.11s versus 17.70/14.65s with whole-model movement. These are short-probe observations, not sustained throughput. Resume restores bound CPU optimizer state and CPU/CUDA RNG; long-run resume throughput remains to be measured.

### Required native-demo timing correction (before scientific residual training)
Official LIBERO native dataset creation stores each observation after executing action at the same index. Confirmed by code inspection and source-only state reconstruction. Converter v3 pairs native observation[i] with action[i+1] and drops the terminal observation. The source file remains the same 50 demonstrations/5882 raw frames; converted training pairs are 5832. Norm statistics are regenerated from exactly that corrected corpus. Source initial-state holdout checks both native first recorded states and first observed successor states.

All earlier same-index SFT checkpoints are now rejected by the alignment contract. Restart corrected alignment from official pretrained pi0, not from those checkpoints. The first corrected pilot remains 100 total updates, preceded by a two-update real-data memory/checkpoint/zero-residual check. This fixes an observation/action integration error; it does not change the frozen task ladder or introduce unseen data.

### Exact corrected normalization and weights
Corrected corpus `local/pld_libero_bowl_v3`; normalizer SHA256 `e04482f55cc670539ade17a753bf48872df9cec573d23a42128ef6e82f9efe78`, with all state/action means, standard deviations and quantiles saved at `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-000/full-cpu-sft-v3-resident-001/checkpoints/2/assets/local/pld_libero_bowl_v3/norm_stats.json`. pi0 uses mean/std normalization; quantiles are retained but not selected. Converted official pretrained weights SHA256 `502ee842b917c553050c5c2f6623a0ed13c1f8ab0bc91a43511f8129e37cf5d2`. Source audit and corrected corpus file hashes are in `outputs/pld_libero/EXP-001/prepare-003/source_audit.json`; normalization producer in the alignment assets directory.

### User-authorized hardware amendment — A100 80 GB
On2026-09-09 the user explicitly instructed us to use the actual A100. The artificial16GiB ceiling is removed for current experiments; configurations now permit the actual80GiB. The frozen task split, seen-only data, source-only selection, full-model trainability, batch 1, LR schedule and residual architecture stay as specified. Next alignment uses the pinned official OpenPI PyTorch train_loop with GPU AdamW, first2 updates to validate memory/checkpoint loading, then the preregistered100-update source-only pilot. This replaces CPU offload for the current experiment; it is closer to upstream training and does not claim RTX 5080 fit. Start from the same official pretrained weights. An upstream final-save off-by-one is corrected in a narrow checkpoint callback so checkpointN meansN completed updates. Native upstream initialization/optimizer/data/forward/backward remain unchanged.

The earlier resident_cpu100-step pilot was prepared but not started. It is superseded by the user-authorized official GPU pilot; its two-step checkpoint and zero-equivalence result remain valid engineering evidence. GPU alignment starts fresh from official pretrained weights to avoid mixing CPU/GPU optimizer histories.

GPU checkpoint retention: save policy weights/statistics/training metadata every 500 completed updates and at the final update. Retain optimizer state only for the latest checkpoint after a newer complete checkpoint exists; keep all policy weights. This bounds local disk use while preserving evaluation checkpoints. Retention actions are logged. The2-step/100-step official GPU runs start fresh from the same verified pretrained weights; no unsupported optimizer resume is implied.

Runtime measurement amendment: scientific commands verify checkpoint/data provenance once before model loading, inside timed run artifacts. Record validation and load durations separately. CUDA allocator peaks are tracked per base-inference, Cal-QL and SAC phase while retaining the overall run peak across resets; these include persistent model allocations. Update logs also record additional transient allocation above their starting footprint. Device-wide samples are persisted incrementally every0.5s so a terminated run retains observed samples.

### Reproducible figures
`scripts/pld/plot_libero.py sft --runs <completed-alignment-run> --output <new-figure-directory>` exports actual training loss with a trailing20-update median and source CSV/provenance. `plot_libero.py transfer --runs <completed-paired-run-dirs...> --output <new-figure-directory>` first recomputes/validates paired results and plots signed equal-task-weight bucket gains per source/training seed. Missing buckets are not zero-filled. Single-training-seed plots do not imply training-seed confidence intervals. Chart contracts and QA notes are stored in EXP-000/audit and the experiment ledger.

### Selected frozen base (source-only decision, 2026-09-09)
The registered3000-update source-only run scored3/10 on validation seeds2000–2009. Freeze `outputs/pld_libero/EXP-001/sft-gpu-v3-3000/checkpoints/pi0_libero_seen_full_torch/EXP-001/3000` and its alignment manifest for the initial residual specialist. No unseen task influenced this selection. Offline collection targets50 successful trajectories within the100 registered training seeds; retain and report the actual successful count if fewer are obtained. Cal-QL/RL smoke must pass on that real buffer before the fixed-budget main residual run.
