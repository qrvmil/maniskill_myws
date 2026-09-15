# D0 base / D1 residual adaptation

Completed 2026-09-15. **The registered experiment did not support competence
extension: D1 gain was 0 percentage points; D0 gain was −18 points.** The D1
improvement gate failed before final evaluation, and every final export preserves
`negative_result_gate_failed`. Implementation: local commit `0c02d01` on
`exp/pld-d0-base-d1-residual`, starting from V3 `da51570`. No remote push.

## 1. Hypothesis

A residual trained only on D1 can extend a frozen D0-aligned pi0 to D1 while
preserving D0 competence. Primary metric: paired D1 success-rate gain. Secondary:
paired D0 retention/backward-transfer gain. D1 measures **trained task adaptation**.
The final metric remains binary LIBERO task success.

## 2. Exact D0/D1 roles

Both tasks belong to `libero_spatial`:

- **D0 / base alignment:** `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- **D1 / residual training:** `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- **Final evaluation:** exactly D0 and D1. The ramekin task is excluded.

The protocol separates `base_alignment_task`, `residual_training_task` and
`evaluation_tasks`, including role-sensitive hashes and replay/selection guards.
The base uses only D0 data; all residual replay, warmup and online RL use D1.

Preregistered fresh blocks: D0 sanity **5000–5049**; D1 training **6000–6099**,
cycled; D1 validation **7000–7049**; final **8000–8049**. Base and residual receive
identical initial states within each task. No final outcome informed selection.
See [configuration](../configs/pld_libero/d0_base_d1_residual.json),
[preregistration](../outputs/pld_libero/V4-preregistration.json) and
[stage rule](../outputs/pld_libero/V4-stage-preregistration.json).

## 3. Base D0 result

Reconstructed from official `gs://openpi-assets/checkpoints/pi0_base`: **LoRA32,
seed 0, batch 8, exactly 3001 optimizer updates**, all 50 official D0 demonstrations,
5832 temporally aligned observation/action pairs and D0-only normalization.
The base was frozen permanently before D1 outcomes were inspected.

Exact checkpoint:
`outputs/pld_libero/V4-sft3001/checkpoints/pi0_libero_seen_lora32/EXP-001/3000`.
The directory index 3000 corresponds to 3001 updates. D0-only
[alignment manifest](../outputs/pld_libero/V4-sft3001/alignment_manifest.json) SHA256:
`a8811e3426985cd39601a721e8bc8689f145f86fa599d46505c54c37322f0f4f`.
It records the checkpoint-file hashes and `training_tasks == [D0]`.

D0 sanity: **32/50 = 64%**, Wilson 95% CI **[50.14%, 75.86%]**.
[Exact-zero audit](../outputs/pld_libero/V4-zero-D0/eval/zero_equivalence.json):
**50/50 paired episodes passed** for actions, physics, images, outcome and length;
zero correction also achieved 32/50. See [base freeze](../outputs/pld_libero/V4-base-freeze.json).

## 4. D1 base feasibility / offline initialization

Frozen-base D1 audit: **0/50** initially and **0/100** after the full collection
block. All episodes lasted 220 steps; none registered a grasp or lift >3 cm.
Mean minimum reach distance was **0.256 m** initially (**0.242 m** over 100);
mean bowl-to-plate progress was **−0.0253 m** (**−0.0200 m** over 100).
The full-100 Wilson SR interval is [0%, 3.70%]. **No successful D1 base trajectory
existed.** The collection artifact's `FAILED` status explicitly records its unmet
successful-replay gate. See [feasibility audit](../outputs/pld_libero/V4-feasibility-summary.json).

The official D1 demonstration fallback supplied **45 genuinely successful,
repeatable reconstructed trajectories / 4444 transitions**; five original action
sequences failed real simulator success and were excluded. Accepted independent
repeats matched physics, proprioception, actions, success and raw images exactly.
**920 actual frozen-base inference calls** on reconstructed observations supplied
base-action chunks with replan interval 5. Human actions never served as base actions.

Native demonstration correspondence failed the strict state/proprioception/image
audit; its [failure evidence](/root/pld-runs/d1-demo-verify-native-002/verification.json)
is retained. The official converter trims the first five actions/states while
regenerating observations; missing controller history is a candidate explanation,
not a proven sole cause. The explicit replacement continuously reexecutes original
D1 actions in the current runtime at render 256/MSAA off, with unchanged source
controller parameters and alignment: post-action-0 observation → action 1 →
post-action-1 observation. **This fallback does not establish native equivalence.**

The verified D1 replay supports 1000 auxiliary full-action Cal-QL updates and the
offline half of online replay. The auxiliary actor and temperature are discarded;
no demonstration BC loss trains the final residual actor, and D1 never updates the
base. [Replay/provenance evidence](../outputs/pld_libero/V4-D1-reexecuted-replay/verification.json);
`offline.npz` SHA256:
`3cc25b1e3289e80af0fdc2e3f8f624455b2de6a477256956c1b551789f23746a`.

Reachability, measured before RL on all 4444 replay transitions: mean
|demo − base| **0.2027**; **87.26% of components** and **37.26% of complete actions**
fit ξ=.5. Most exact expert commands were therefore outside the single-step residual
bound; alternative successful trajectories remain possible. **ξ stayed .5.**

| Dimension | Mean absolute required correction | Components within .5 |
|---|---:|---:|
| x | .2860 | 83.17% |
| y | .3926 | 66.22% |
| z | .3549 | 73.40% |
| rotation x | .0303 | 100% |
| rotation y | .0608 | 100% |
| rotation z | .0487 | 100% |
| gripper | .2459 | 88.05% |

Required saturation: **12.74%** of components. Preclip base output exceeded bounds
in **6.70%** of components (maximum magnitude 1.0778). Projecting the required edit
to ξ=.5 caused 0% additional controller clipping; action mismatch MAE was .05093,
maximum 1.5. Full per-dimension quantiles: [reachability audit](../outputs/pld_libero/V4-D1-reexecuted-replay/reachability.json).

## 5. D1 residual training curve

New specialist, one seed. V3 SAC retained: batch 256, replay 250k, offline:online
50:50, γ=.99, actor update interval 2, pretrained SERL ResNet10 with frozen trunk, shared visual
representation, learned state-independent std, unit residual density with separate
physical ξ=.5, mean-Q actor, min-Q TD target, softplus temperature and 2000-step
optimizer warmup. **Probe fraction 0; sparse success reward.**

Completed 1000 auxiliary Cal-QL updates, then **100 base-only D1 warmup episodes /
22,000 SAC updates**, including 11,000 actor updates. Warmup exactly reproduced the
initial 100 base-audit trajectories; it is not another independent baseline sample.
Critic, actor head/std and temperature changed as required. Active training then
completed **50,160 steps / 228 episodes / 0 successes**. All 72,160 online updates
were finite and used exactly 50:50 replay.

Every validation used the complete same 50 paired D1 seeds:

| Milestone | Actual active steps | Base success | Residual success | Rescue / harm | Mean executed correction |
|---|---:|---:|---:|---:|---:|
| 5k | 5060 | 0/50 | 0/50 | 0 / 0 | .008875 |
| 10k | 10120 | 0/50 | 0/50 | 0 / 0 | .006366 |
| 25k | 25080 | 0/50 | 0/50 | 0 / 0 | .011879 |
| 50k | 50160 | 0/50 | 0/50 | 0 / 0 | .032322 |

The preregistered continuation criterion required 50k D1 validation SR to strictly
exceed 25k SR. It did not, so training ended without 100k. Episode-boundary overshoot
was <220 steps. The four validations are repeated measurements on 50 seeds,
not 200 independent tests.

[Training curve, PNG](../outputs/pld_libero/V4-training-figures/training_curve.png)
([PDF](../outputs/pld_libero/V4-training-figures/training_curve.pdf));
[full diagnostics, PNG](../outputs/pld_libero/V4-training-figures/training_diagnostics.png)
([PDF](../outputs/pld_libero/V4-training-figures/training_diagnostics.pdf)).
Curves show training/validation SR, residual and per-dimension magnitude, std,
temperature, Q estimates/margins, OTF selection, Bellman/Cal-QL losses, clipping and
replay mixture. Episode curves use trailing windows of up to 25; validation error
bars are pointwise Wilson intervals. [Figure provenance](../outputs/pld_libero/V4-training-figures/figure_provenance.json)
records source hashes and aggregation rules.

## 6. Selected D1 specialist

Selected `outputs/pld_libero/V4-D1-training/checkpoints/residual_step_32120.pt`:
**10,120 active steps** plus 22,000 warmup steps. SHA256:
`97b09205b53941df0284c7b5d5d70ad53035bc65987a3632b5a94c34af23905c`.
All candidates tied at 0/50 success; this checkpoint had the smallest mean executed
correction (.006366), following the preregistered tie rule.

**D1 improvement gate failed:** gain 0 points, rescue/harm 0/0. Deployment stayed
**deterministic_actor**. [Selection manifest](../outputs/pld_libero/V4-selected-D1.json)
SHA256: `0ce45f7983c9857bad8bf4ff3222206c4fb01ac82de928754a3f575c753bf3e4`.
[Final freeze](../outputs/pld_libero/V4-final-freeze.json) was written at
**2026-09-15 02:10:49 UTC**, before either final run began. It binds base/residual
hashes, deployment and configuration file SHA256:
`b1f376cb628098e8d162f4c6da42809ae9142d08ea1a1129ee976635080a2070`.
No D0 residual outcome informed checkpoint, duration, scale or deployment choices.

## 7. Final D0/D1 paired table

Fresh seeds 8000–8049; 50 exact initial-state pairs per task:

| Task | Base SR | Base + D1 residual SR | ΔSR, percentage points | Rescue | Harm | N |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 41/50 = **82%** | 32/50 = **64%** | **−18** | 3 | 12 | 50 |
| D1 | 0/50 = **0%** | 0/50 = **0%** | **0** | 0 | 0 | 50 |

**D0 gain 95% CI: [−32, −4] points**, paired percentile bootstrap, 10,000 resamples,
seed 0. Marginal Wilson SR intervals: base [69.20%, 90.23%], residual [50.14%, 75.86%].

**D1 gain 95% bound: [−5.82, +5.82] points.** With zero observed discordance, the
raw bootstrap [0,0] is degenerate. We instead use the exact 95% upper bound
`1 − .05^(1/50)` on discordance probability to bound the gain. Each D1 policy's
Wilson SR interval is [0%, 7.13%]. Intervals concern evaluation episodes under this
one trained specialist; they do not measure variation across training seeds.

The 64% D0 sanity rate used different seeds. Retention must be compared against
the **paired final base rate of 82%**, not the earlier sanity result.
Machine-readable [JSON](../outputs/pld_libero/V4-final-results/summary.json),
[CSV](../outputs/pld_libero/V4-final-results/tasks.csv), and independent
[final audit](../outputs/pld_libero/V4-final-audit.json) reproduce the table and intervals.

Eight verified MP4s in `outputs/pld_libero/V4-final-D0/eval/` provide separate
base/residual videos for harm seeds **8000, 8001** and rescue seeds **8037, 8041**.
D1 had no rescue/harm cases, so no category videos exist.
[Video verification](../outputs/pld_libero/V4-video-verification.json) records paths,
labels, hashes and successful full decoding.

## 8. Interpretation

**The hypothesis failed in this registered configuration and training seed.**
The residual did not extend the frozen base to D1, and it interfered with D0:
12 previously successful episodes became failures, versus three rescues. The
paired D0 interval excludes zero. This result does not establish that residual
adaptation is impossible under other settings.

Diagnostics constrain the explanation. The base produced no D1 online successes;
the verified successful offline trajectories did not lead to online success.
Only 37.26% of exact expert actions fit the primary residual bound. Base-inclusive
OTF selected the base on **63.1%** of active training steps despite zero probing.
Training mean executed correction was .0861, while deterministic validation edits
were much smaller. At the end of active training, temperature was approximately **3.90e−4** and
pre-tanh std was **.737–.818** across dimensions. Critic action contrasts were small;
these observations alone do not prove a particular optimization failure.

Post-hoc final mean executed correction was **.006255 on D0** and **.006394 on D1**.
Small action magnitude did not ensure retention. On D0 both policies lifted in
50/50 episodes, but task success differed; on D1 neither registered any grasp or
lift >3 cm. [Post-hoc diagnostics](../outputs/pld_libero/V4-posthoc-policy-summary.json)
are descriptive and did not change any model or selection decision.

The pinned LIBERO reward source returns sparse success; its `reward_shaping` flag
does not add a dense term in that implementation. Simulator reach/grasp/lift/plate
progress is available for diagnostics. [Reward-source audit](../outputs/pld_libero/V4-reward-source-audit.json)
is preserved. No shaping, ξ=1.0, probing or post-final tuning ablation was run.

## 9. Deviations from PLD

Intentional D0-only LoRA alignment followed by D1-only residual adaptation;
probe fraction 0; fixed deterministic deployment; explicit D1 action-reexecution
fallback for critic/offline replay initialization. No D1 base updates, residual
actor demonstration BC, second SFT, distillation or hybrid-data collection.

Inherited V3 choices: synchronous learner, base-inclusive OTF training,
finite-horizon terminal masks, raw proprioception and no image augmentation.
The 2000-step optimizer warmup follows the SERL SAC default; its DrQ factory
disables it. AdamW decay .01 is an unpublished-setting choice. Independent final
Q projections, Xavier visual projection initialization and current-state entropy
for temperature differ from the referenced SERL path. Auxiliary Cal-QL is a
reviewed port, not bitwise reference parity. Fresh generated resets differ from
benchmark demo-initial-state evaluation. One training seed and the registered
50k/conditional-100k budget limit the inference. See [V3 audit](RESIDUAL_V3_AUDIT.md).

**Runtime and verification.** NVIDIA A100-SXM4-80GB, Torch 2.7.1+cu128, JAX 0.5.3,
robosuite 1.4.1, MuJoCo 3.2.7 and pinned LIBERO/OpenPI revisions.

| Stage | Wall time | Sampled whole-device peak VRAM |
|---|---:|---:|
| D0 SFT | 116.83 min | 33,673 MiB |
| D0 zero/base audit | 16.14 min | 9,352 MiB |
| D1 base collection | 23.45 min | 10,090 MiB |
| Verified D1 replay | 13.89 min | 8,631 MiB |
| Residual training, including warmup/staging | **434.23 min / 7.24 h** | 21,448 MiB |
| D1 validation 5k / 10k / 25k / 50k | 34.15 / 34.02 / 33.80 / 25.65 min | 21,448 MiB |
| Final D1 | 26.59 min | 9,676 MiB |
| Final D0, including videos | 18.86 min | 9,676 MiB |

Training wall time includes initialization, warmup, serialization and stage waits.
Validations overlap it; these durations must not be added as end-to-end time.
The maximum measured usage was **33,673 MiB = 32.88 GiB** during SFT; shared-device
peaks during RL include concurrent validation. Resource metadata is in the final audit.

Pretraining regression suite, including historical V2/V3: **112 passed, 6 optional skipped**. Real MuJoCo
zero/reset integration and pinned SERL feature plus distribution/shared-gradient
parity passed. Independent reviews covered role boundaries, replay provenance,
staging/selection and final export controls. Scientific source stayed at `0c02d01`
through training and final evaluation. The final independent audit passed exact
pairing, frozen hashes, all four D1 selection candidates, every logged numeric
update and replay mixture. Both figures were visually checked; all eight videos
fully decoded and their labels matched recorded outcomes.
