# PLD → LIBERO V3

The D0-selected specialist improves validation from **32/50 (64%) to40/50 (80%)**. Its checkpoint and deterministic deployment were frozen on2026-09-11 at03:55UTC, before fresh final outcomes. D0–D3 evaluation is running. This is an exploratory direct-residual experiment, without distillation or target training.

## 1. Changes from V2

See the [reference audit](RESIDUAL_V3_AUDIT.md), [preregistered plan](plans/2026-09-10-pld-libero-v3.md) and [training config](../configs/pld_libero/anchor_bowl_v3.json).

- Added the missing per-active-episode base probing. Draw an integer prefix uniformly from0 through floor(0.3×horizon), advance the real base cache/RNG, retain the original episode horizon, and exclude every prefix transition from residual replay. The0.3 upper fraction is our declared setting; PLD does not publish the exact training range.
- Added SERL uniform learned state-independent std, mean-Q actor/min-Q target, unit-tanh density with separate physical scaleξ=0.5, softplus temperature, and2000-gradient-step online actor/critic LR warmup.
- Shared actor/Q1/Q2 visual representation; actor loss stops encoder gradients, critic learns pooling/projection. The pretrained convolution trunk **remains frozen**, matching the actual SERL pretrained path. Target critics share a separate target encoder.
- Cal-QL uses an auxiliary full-action offline actor; its head/temperature are discarded before online learning. A fixed-random-proposal initialization was retained as a diagnostic control.
- Corrected the V2 audit: **PLD Algorithm1 updates the actor during base-only warmup**. Primary V3 follows that schedule. The actor-frozen run is a separately retained control, not the primary reference reproduction. This correction preceded all active interaction and residual validation.

Primary base: fixed source-only LoRA32 **SFT3001**, chosen before residual results; no new SFT or target normalization. Strong-base SFT4000 was not rerun. Batch256, replay250k, offline fraction0.5, Cal-QL1000updates,100base warmup episodes, one-edit-plus-base OTF, one training seed. Scale schedule implemented but not run.

Checks: **80tests passed**, two unrelated ManiSkill tests skipped for absent dependencies; LIBERO and actual SERL/Distrax integration checks passed. Base/exact-zero matched **50/50** actions, outcomes, lengths, simulator states and images. Base collection yielded50successes/72attempts,5910transitions. Frozen-warmup control preserved actor head/std and alpha exactly; primary warmup reproduced the same100base trajectories and69successes while updating critic14,889times and actor7,444times. All50,023active updates and replay/probing accounting checks passed.

## 2. D0 training curve

Paired interim validation uses seeds2000–2019. These repeated checkpoints are not independent replications.

| ACTIVE steps | Base SR | Deterministic SR | OTF SR | Δdet | ΔOTF |
|---:|---:|---:|---:|---:|---:|
| 5,103 | 14/20 (70%) | 10/20 (50%) | 10/20 (50%) | −20pp | −20pp |
| 10,095 | 14/20 (70%) | 15/20 (75%) | 9/20 (45%) | +5pp | −25pp |
| 25,000 | 14/20 (70%) | 15/20 (75%) | 7/20 (35%) | +5pp | −35pp |
| 50,023 | 14/20 (70%) | 2/20 (10%) | 3/20 (15%) | −60pp | −55pp |

Training completed the required≥50k ACTIVE stage:64,912online replay transitions including14,889warmup,78,268simulator steps including probing. It was not stopped for the5k drop. At50k the deterministic correction grew to0.100 from0.017 at25k, OTF correction to0.154, and temperature fell to0.000861. Late degradation and weak critic ranking give no reason to extend to100k. This is consistent with critic exploitation, not proof of a newly identified implementation bug. The negative late checkpoints remain available.

## 3. Selected D0 specialist

Full paired source validation uses2000–2049. The50k modes failed the recorded interim finalist screen; only the10k/25k deterministic finalists were expanded to50. Selection maximizes D0 validation SR, then prefers smaller correction and earlier checkpoint. No final or target outcomes participated.

| Full validation candidate | Base | Residual | Gain | Rescue / harm | Paired bootstrap95% gain interval |
|---|---:|---:|---:|---:|---:|
| 10,095 deterministic | 32/50 (64%) | 34/50 (68%) | +4pp | 10 /8 | [−12,+20]pp |
| **25,000 deterministic** | **32/50 (64%)** | **40/50 (80%)** | **+16pp** | **14 /6** | **[0,+32]pp** |

Both50-seed runs reproduce the base oracle and their repeated first20 base/residual trajectories exactly. N=50, not70. The selected point estimate meets the requested SR criterion; **statistical significance and reliable multi-seed improvement are not claimed**.

Frozen deployment: `V3-moderate-A-reference/checkpoints/residual_step_39889.pt`, ACTIVE25,000, SHA256 `7057c032ca294b1cf0b22ab9aa6e3938f0392026378d84a9ed8027d734246bcf`; mode `deterministic_actor`; mean executed validation correction0.01738. Immutable selection and base/config hashes: `outputs/pld_libero/V3-selection/{source_selection,deployment}.json`.

## 4. Frozen D0–D3 transfer

Evaluation running with the [frozen transfer config](../configs/pld_libero/anchor_bowl_v3_transfer.json). It differs from training config only by opening the authorized transfer gate. All six tasks use the same base, residual, deployment and paired fresh seeds4000–4049, registered before outcomes. First20 per task, then full50 if the first stage takes≤90minutes; extension depends only on runtime. No D4/D5 evaluation.

## 5. Important caveats

Counterfactual diagnostic: eight states from two successful and two failed D0 training trajectories; one candidate action followed by frozen-base continuation. Exact reconstruction and common candidate rollout hashes were verified. This estimates base-continuation outcomes, not unbiased residual-policy Q values.

| Critic | Success-ranking accuracy, non-tied pairs | Harmful edits preferred | Rescue edits rejected | Discounted-return ranking |
|---|---:|---:|---:|---:|
| Auxiliary Cal-QL | 9/16 | 0/4 | 2/2 | 22/62 |
| Fixed-random Cal-QL control | 9/14 | 0/3 | 1/1 | 32/61 |
| Primary after warmup | 10/25 | 2/5 | 2/2 | 29/64 |
| Primary ACTIVE10,095 | 6/25 | 3/5 | 2/2 | 23/62 |

Actor-dependent candidates differ across rows. On identical base/±small actions, discounted ranking is5/12,6/12,6/12,5/12 respectively: no demonstrated Cal-QL initialization advantage. Mean edit-minus-base Q margin after warmup/10k is−0.00222/−0.00116. Only four independent trajectories make uncertainty large; the10k discounted-ranking trajectory-bootstrap interval is[0,0.536]. **OTF ranking remains unreliable**, consistent with its poor validation. Candidate count was never increased. Calibration plots, per-state outcomes and paired rescue/harm videos are retained in artifacts.

Remaining fidelity limits: unpublished exact PLD integration/probing range/offline budget; source-only LoRA alignment; chosen SERL SAC LR warmup (DrQ factory differs); PLD AdamW decay unspecified, ours0.01; raw proprioception, no crop/dropout, independent final Q readouts, Xavier projections, current-state temperature sampling and synchronous update ordering. Auxiliary Cal-QL uses gradient clipping unlike its standalone reference. One training seed and selection on D0 validation limit inference; final transfer is exploratory.

Hardware: A100-SXM4-80GB; Torch2.7.1+cu128, JAX0.5.3, CUDA12.8; cgroup RAM241.7GiB. Main run elapsed about6.49h including pauses/concurrent validation; learner update compute3.85h. Peak Torch allocation1.97GiB/reservation2.27GiB; whole-device peak44.21GiB with one concurrent evaluator, including JAX allocation. Separate multi-process audit/evaluation peaks are not standalone learner memory.

Raw commands, hashes, tests, training curves, losses, per-action diagnostics and timing: `outputs/pld_libero/V3-*`. Historical V2 artifacts remain unchanged. Branch `fix/pld-libero-residual-v3`; no push.
