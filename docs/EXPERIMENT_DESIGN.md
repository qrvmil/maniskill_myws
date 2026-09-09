# EXPERIMENT_DESIGN — PLD LIBERO residual transfer

Specification frozen before policy results, 2026-09-09. Machine-readable split:
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

| Distance | Suite | Exact task name | Object | Destination | Relation | Primitive | Skill subgoals | Step cap | Language/layout changes and reason |
|---|---|---|---|---|---|---|---|---:|---|
| D0 | libero_spatial | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | black bowl | plate | center → on plate | pick/place | 1 | 220 | anchor; two bowls, spatial grounding |
| D1 | libero_spatial | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | black bowl | plate | next to plate → on plate | pick/place | 1 | 220 | same primitive/object/destination; source location and distractor placement vary |
| D1 | libero_spatial | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | black bowl | plate | next to ramekin → on plate | pick/place | 1 | 220 | same primitive/object/destination; spatial phrase and bowl placement vary |
| D2 | libero_goal | put_the_bowl_on_the_stove | black bowl | stove | on stove | pick/place | 1 | 300 | destination semantics and layout change |
| D2 | libero_object | pick_up_the_alphabet_soup_and_place_it_in_the_basket | alphabet soup | basket | in basket | pick/place | 1 | 280 | object, destination, language and layout change |
| D3 | libero_goal | open_the_top_drawer_and_put_the_bowl_inside | bowl + drawer | top drawer | inside | open then pick/place | 2 | 300 | opening prerequisite; BDDL has one final In predicate, not two success predicates |
| D4 | libero_goal | open_the_middle_drawer_of_the_cabinet | middle drawer | cabinet | open | articulation | 1 | 300 | new primitive and goal language; shared tabletop assets but changed layout |
| D4 | libero_goal | push_the_plate_to_the_front_of_the_stove | plate | stove front | front of | push | 1 | 300 | contact manipulation without bowl grasp; goal/layout change |
| D5 | libero_10 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | bowl + drawer | bottom drawer | inside + closed | pick/place then close | 2 | 520 | kitchen scene; initially open drawer; In AND Close goals, longer evaluation budget |

## Leakage rules
Unseen definitions may be read to specify distance and construct evaluation environments. Unseen demonstrations, normalization statistics, rewards, rollouts and success rates must not influence SFT, offline replay, RL, hyperparameters or checkpoint selection. No full-LIBERO aligned checkpoint in primary runs. Training/collection must reject a non-source task. Manifest records source ID, demo hashes, statistics hashes, pretrained checkpoint and aligned checkpoint. A missing alignment manifest blocks scientific rollout commands. Unseen evaluation begins only after fixing the checkpoint. Additional anchors require separate specialists and preregistered ladders.

## Base-policy alignment
Preferred pretrained checkpoint: `gs://openpi-assets/checkpoints/pi0_base` (not pi0/pi05_libero). Official OpenPI LIBERO LeRobot format; source-only physical subset with source-only normalization.
Raw action: 7D normalized robosuite OSC_POSE input (delta xyz, delta axis-angle, gripper), [-1,1]. Controller maps to physical deltas; do not apply joint-position conventions from ManiSkill. State: eef xyz + quaternion converted to axis-angle + two gripper qpos = 8D. Cameras ordered agentview, robot0_eye_in_hand; render 256, rotate both axes 180 degrees per official evaluation, resize/pad to 224. Model adds masked third camera. Language comes from benchmark task.language.
Use `extra_delta_transform=False` for newly trained checkpoint because LIBERO actions already are deltas. This differs from upstream legacy pi0_libero config with extra delta enabled; training and inference must agree. OpenPI owns standardization/inverse-standardization; residual operates in bounded environment units AFTER inverse transform.
Full SFT first: bf16, batch 1, no EMA, gradient checkpointing, single GPU; fixed source-only validation. Proposed 3000 SFT steps for exploratory run, save every 500, choose using source-only validation. Actual demo count, normalizer and measured parameters are recorded below. Official full-SFT estimate >70 GB; official LoRA estimate >22.5 GB. No assumption either fits 16 GB. A real limited-memory probe and official alternative audit are required before a deviation. Do not use A100 capacity to claim 5080 feasibility.

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
Target one RTX 5080 16 GB; actual attached device is NVIDIA A100 80GB PCIe, driver 595.71.05, torch 2.11.0+cu128, CUDA wheel 12.8. Measurements here are A100 results; a 16 GiB software allocation ceiling is only a feasibility proxy, not Blackwell validation. One environment, CPU replay, sequential rollout/update, one base model. Measure inference, updates and combined peak allocated/reserved and device-wide VRAM, plus wall times. Workspace is not volume-backed.

## Deviations from PLD paper
| Paper | Our implementation | Reason | Expected consequence |
|---|---|---|---|
| Three-stage PLD including distillation | Stop after specialist acquisition; directly test frozen residual | Research question | Does not reproduce published distilled generalization numbers |
| Many source specialists/data collection scale | One anchor, one environment, one seed initially | 16 GB local constraint | Higher variance and lower throughput |
| Pretrained ResNetV1-10 visual residual | Existing randomly initialized channel-stacked GroupNorm ResNet-like encoder | Reuse unofficial reproduction; no matching pretrained weights bundled | Material representation/pretraining mismatch, potentially weaker transfer |
| Successful frozen-base offline rollouts | Same requirement; new LIBERO buffer, no legacy proxy | Avoid fidelity regression | Can block if aligned base has no successes |
| Paper sparse reward and training hyperparameters | Sparse terminal reward (matches paper main runs); inherited SAC/Cal-QL settings | Explicit practical baseline | Optimization may differ substantially |
| OTF defaults to 1 residual sample around fixed base | 1 residual sample plus explicit unedited base candidate, min-Q argmax and hard-max target | Closest existing reproduction path | Fixed-base sampling matches paper; extra base candidate/entropy treatment remain reproduction choices |
| Paper-scale batches/update ratio | Batch 8, 1 update/step initially | Single 16 GB design | Learning speed/stability may differ |
| Standard benchmark evaluation init files | Disjoint seeded fresh initializations | Stronger same-task holdout from demos | Not directly comparable to official benchmark SR |
| Paper appendix uses rank-32 LoRA on 8 L40s | Full pi0 forward/backward on GPU, AdamW state/update on CPU; batch 1, checkpointing, no EMA | GPU AdamW and official LoRA both measured OOM | Preserves full-model trainability; slower transfers/CPU optimizer, bf16 rounding may differ from GPU optimizer |
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

## Pre-training amendments and measured feasibility (2026-09-09)
- Native official source data verified: 50 demonstrations, 5882 frames, file SHA256 `75ede0cf5fbfc925093671b55032ad80f1b1f1cf35442ec6c663460d37d3e0b3`. Only this task file downloaded, from `yifengzhu-hf/LIBERO-datasets`, the mirror named by LIBERO download_utils. HDF5 BDDL metadata must match the source before conversion.
- Native HDF5 cameras are 128x128, not OpenPI example RLDS 256x256. We rotate both axes for both native demonstration cameras and runtime cameras, then use OpenPI resize/pad to 224. Runtime currently renders 256px; downsampled visual RL is 128px. Retaining different native/render resolutions is an explicit image-resolution deviation. No no-op filtering initially (all 5882 frames); control frequency and LeRobot FPS are 20. Official modified RLDS example uses 10 FPS and no-op-filtered data.
- Full-model memory probe: real official pi0 architecture, random weights/synthetic inputs (no alignment claim), 3501372176 parameters, 7011414880 parameter bytes. Batch 1, official selective bf16, gradient checkpointing, no EMA; actual optimizer-state OOM under 16 GiB PyTorch ceiling after forward/backward. EXP-000/full_sft_probe/run-002 contains exact failure. First probe failed on BHWC vs BCHW test-input layout; both preserved.
- Closest official supported fallback selected for feasibility testing: `pi0_libero_low_mem_finetune`, both Gemma LoRA variants, batch 1, no EMA, unchanged official freeze filter. The filter freezes non-LoRA LLM parameters; vision/projections remain trainable. This is a material alignment deviation from full SFT, not evidence it fits yet. JAX allocator fraction will be constrained to an approximately 16 GiB pool; that pool is not a whole-process VRAM hard limit. Do not infer fit until device-wide measurements confirm it.
- Soft physics resets (`hard_reset=False`) preserve renderer context. Hard resets produced 1-LSB RGB differences for otherwise identical physics/action trajectories. A short contract run passed, but longer repeated runs still exposed 1–2 intensity RGB differences despite exact physics/actions; renderer determinism investigation remains open. This does not reuse robot/object placements: each reset samples them using its registered seed.

## Pre-inference guard amendments (2026-09-09)
- Both GPU AdamW full SFT and official LoRA failed the constrained-memory probes. Selected closest feasible path is **full-model CPU AdamW**, not LoRA: official PI0Pytorch forward, data transforms, selective bf16, gradient checkpointing and AdamW hyperparameters; gradients clipped on GPU and the single model moved to CPU for the update. No extra parameters are frozen. Synthetic probe peaked at 13626453504 allocated bytes (12.69 GiB); real-data SFT remains a separate gate. No gradient accumulation (batch 1).
- Converted dataset bytes, repo ID, dataset root, normalizer producer and bytes, and aligned checkpoint bytes are bound to provenance manifests. Normalization is re-run after adding producer provenance. Base action cadence and render/image settings are bound to zero-check, offline-buffer and specialist provenance. Resume binds source corpus, normalizer, pretrained weights, seed and optimizer configuration.
- Numerical aligned-policy zero-equivalence threshold fixed before inference: action/full simulator-state max absolute error <=1e-5, identical reset hash, episode length and success; RGB <=1 uint8 intensity. Longer dummy runs that fail RGB remain failed; renderer fixes must precede this gate. This is numerical equivalence, not bitwise identity. Dummy action and physics comparisons remain exact.
- OpenPI's converter initializes an unused expert lm_head absent from JAX pi0; independent conversions differ only there. Functional checkpoint tensors are checked separately in conversion verification. Checkpoint hashes record the actual full output including this unused tensor.
- Eager PyTorch inference is used (`TORCH_COMPILE_DISABLE=1`) to avoid max-autotune compilation caches competing with the 16 GiB budget. This changes throughput, not the action architecture. The frozen base has requires_grad=False explicitly.
- Current pinned OpenPI environment uses torch 2.7.1+cu126 on A100. It must use a cu128-or-newer wheel on RTX 5080; current runtime is not Blackwell validation.

## Runnable commands
Run from the repository root. `scripts/pld/setup_libero.sh` provisions the pinned simulator/OpenPI environment with CUDA 12.8 wheels. `scripts/pld/libero_python.sh` selects this environment, source paths, EGL, CPU JAX and eager PyTorch. Every output directory must be new; never overwrite prior runs. The commands below are templates for pending stages, not evidence they ran.

```sh
scripts/pld/libero_python.sh scripts/pld/prepare_pi0_libero.py --output /workspace/checkpoints/pi0_base_pytorch_new
scripts/pld/libero_python.sh scripts/pld/align_libero.py prepare --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_new --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_new --output outputs/pld_libero/EXP-001/prepare-new
scripts/pld/libero_python.sh scripts/pld/align_libero.py norm --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --output outputs/pld_libero/EXP-001/norm-new
scripts/pld/libero_python.sh scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --steps 3000 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-001/sft-3000
scripts/pld/libero_python.sh scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-001/zero-001
scripts/pld/libero_python.sh scripts/pld/run_libero.py base --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --episodes 50 --output outputs/pld_libero/EXP-001/base-001
scripts/pld/libero_python.sh scripts/pld/run_libero.py collect --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-001/eval/zero_equivalence.json --successes 50 --max-attempts 100 --output outputs/pld_libero/EXP-001/offline-001
scripts/pld/libero_python.sh scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_rl_smoke.json --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-001/eval/zero_equivalence.json --offline-buffer outputs/pld_libero/EXP-001/offline-001/offline.npz --output outputs/pld_libero/EXP-002/rl-smoke-001
scripts/pld/libero_python.sh scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-001/eval/zero_equivalence.json --offline-buffer outputs/pld_libero/EXP-001/offline-001/offline.npz --output outputs/pld_libero/EXP-002/source-otf-001
scripts/pld/libero_python.sh scripts/pld/run_libero.py eval --config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest outputs/pld_libero/EXP-001/sft-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-001/eval/zero_equivalence.json --checkpoint outputs/pld_libero/EXP-002/source-otf-001/checkpoints/residual_step_50000.pt --distance D1 --episodes 50 --output outputs/pld_libero/EXP-003/paired-001
```
Use D0 for source paired evaluation under EXP-002, and D2/D3/D4/D5 with EXP-004/005/006/007. The same frozen specialist is used throughout. Default config is A (no OTF); separate OTF config uses paper-default residual candidate count 1 and warmup 100 episodes; debug A defaults to 5 warmup episodes. The short scientific RL config still requires a real successful aligned-base buffer.

### Renderer amendment
Disabling offscreen MSAA (`offsamples=0`, default upstream 4) eliminated RGB mismatch in a 220-step diagnostic and the subsequent six-episode smoke. This alters anti-aliasing relative to native demonstration images, and is a documented observation-rendering deviation. Physics, controller and task definitions are unchanged. The conservative numerical zero gate remains mandatory for each aligned checkpoint.

### Final paper audit correction before residual training
[PLD Appendix B.2/C](https://arxiv.org/html/2511.00091v1#A2.SS2) explicitly samples residuals around a **fixed** base action and defaults to OTF=1; holding the base fixed is therefore not itself a deviation. The initial draft overstated that difference. Main B now uses one residual candidate plus the reproduction's explicit base candidate, and 100 base warmup episodes. A remains the smaller debugging baseline. The paper uses pretrained visual encoders; our inherited encoder starts randomly, a material deviation. Appendix C also describes LoRA SFT; this project's full-model alignment follows the user's requested fidelity preference after measured memory probes, not a claim that the paper required full SFT. These changes were registered before any residual training or unseen evaluation.

### Actual full-model alignment feasibility
EXP-000/full-cpu-sft-003 completed two real source-only updates on torch 2.7.1+cu128, with peak allocated 13626460160 bytes and sampled total device usage 13739 MiB on A100. The saved model has no additional frozen parameters. Official CPU initialization is retained because meta/to_empty failed to materialize nonpersistent positional buffers. A two-step checkpoint is an engineering fixture, not evidence of adequate alignment. At the observed second-step time of 17.0 seconds, 3000 steps project to roughly 14 hours before checkpoint overhead; this is a runtime projection, not a measured run. Longer alignment budgets will be declared before their source-only evaluation.
