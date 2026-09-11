# PLD → LIBERO V3

The frozen specialist improves fresh final D0 from **31/50 (62%) to 36/50 (72%): +10 pp**, with 12 rescues/7 lost base successes and paired bootstrap 95% interval[−6,+26]pp. It was selected using D0 validation only (32/50→40/50) and frozen on 2026-09-11 at 03:55 UTC, before fresh final outcomes. D0–D3 evaluation is complete at N=50 per task. No distillation or target training; statistical significance is not claimed.

## 1. Changes from V2

See the [reference audit](RESIDUAL_V3_AUDIT.md), [preregistered plan](plans/2026-09-10-pld-libero-v3.md) and [training config](../configs/pld_libero/anchor_bowl_v3.json).

- Added the missing per-active-episode base probing. Draw an integer prefix uniformly from 0 through floor(0.3×horizon), advance the real base cache/RNG, retain the original episode horizon, and exclude every prefix transition from residual replay. The 0.3 upper fraction is our declared setting; PLD does not publish the exact training range.
- Added SERL uniform learned state-independent std, mean-Q actor/min-Q target, unit-tanh density with separate physical scaleξ=0.5, softplus temperature, and 2000-gradient-step online actor/critic LR warmup.
- Shared actor/Q1/Q2 visual representation; actor loss stops encoder gradients, critic learns pooling/projection. The pretrained convolution trunk **remains frozen**, matching the actual SERL pretrained path. Target critics share a separate target encoder.
- Cal-QL uses an auxiliary full-action offline actor; its head/temperature are discarded before online learning. A fixed-random-proposal initialization was retained as a diagnostic control.
- Corrected the V2 audit: **PLD Algorithm 1 updates the actor during base-only warmup**. Primary V3 follows that schedule. The actor-frozen run is a separately retained control, not the primary reference reproduction. This correction preceded all active interaction and residual validation.

Primary base: fixed source-only LoRA32 **SFT3001**, chosen before residual results; no new SFT or target normalization. Strong-base SFT4000 was not rerun. Batch 256, replay 250k, offline fraction 0.5, Cal-QL 1000 updates, 100 base warmup episodes, one-edit-plus-base OTF, one training seed. Scale schedule implemented but not run.

Checks: **81 tests passed**, two unrelated ManiSkill tests skipped for absent dependencies; LIBERO and actual SERL/Distrax integration checks passed. Base/exact-zero matched **50/50** actions, outcomes, lengths, simulator states and images. Base collection yielded 50 successes/72 attempts,5910 transitions. Frozen-warmup control preserved actor head/std and alpha exactly; primary warmup reproduced the same 100 base trajectories and 69 successes while updating critic 14,889 times and actor 7,444 times. All 50,023 active updates and replay/probing accounting checks passed.

## 2. D0 training curve

Paired interim validation uses seeds 2000–2019. These repeated checkpoints are not independent replications.

| ACTIVE steps | Base SR | Deterministic SR | OTF SR | Δdet | ΔOTF |
|---:|---:|---:|---:|---:|---:|
| 5,103 | 14/20 (70%) | 10/20 (50%) | 10/20 (50%) | −20 pp | −20 pp |
| 10,095 | 14/20 (70%) | 15/20 (75%) | 9/20 (45%) | +5 pp | −25 pp |
| 25,000 | 14/20 (70%) | 15/20 (75%) | 7/20 (35%) | +5 pp | −35 pp |
| 50,023 | 14/20 (70%) | 2/20 (10%) | 3/20 (15%) | −60 pp | −55 pp |

Training completed the required≥50k ACTIVE stage:64,912 online replay transitions including 14,889 warmup,78,268 simulator steps including probing. It was not stopped for the 5k drop. At 50k the deterministic correction grew to 0.100 from 0.017 at 25k, OTF correction to 0.154, and temperature fell to 0.000861. Late degradation and weak critic ranking give no reason to extend to 100k. This is consistent with critic exploitation, not proof of a newly identified implementation bug. The negative late checkpoints remain available.

## 3. Selected D0 specialist

Full paired source validation uses 2000–2049. The 50k modes failed the recorded interim finalist screen; only the 10k/25k deterministic finalists were expanded to 50. Selection maximizes D0 validation SR, then prefers smaller correction and earlier checkpoint. No final or target outcomes participated.

| Full validation candidate | Base | Residual | Gain | Rescue / harm | Paired bootstrap 95% gain interval |
|---|---:|---:|---:|---:|---:|
| 10,095 deterministic | 32/50 (64%) | 34/50 (68%) | +4 pp | 10 /8 | [−12,+20]pp |
| **25,000 deterministic** | **32/50 (64%)** | **40/50 (80%)** | **+16 pp** | **14 /6** | **[0,+32]pp** |

Both 50-seed runs reproduce the base oracle and their repeated first 20 base/residual trajectories exactly. N=50, not 70. The selected point estimate meets the requested SR criterion; **statistical significance and reliable multi-seed improvement are not claimed**.

Frozen deployment: `V3-moderate-A-reference/checkpoints/residual_step_39889.pt`, ACTIVE25,000, SHA256 `7057c032ca294b1cf0b22ab9aa6e3938f0392026378d84a9ed8027d734246bcf`; mode `deterministic_actor`; mean executed validation correction 0.01738. Immutable selection and base/config hashes: `outputs/pld_libero/V3-selection/{source_selection,deployment}.json`.

## 4. Frozen D0–D3 transfer

The [frozen transfer config](../configs/pld_libero/anchor_bowl_v3_transfer.json) differs from training config only by opening the authorized transfer gate. All six tasks use the same base, residual, deployment and paired fresh seeds 4000–4049, registered before outcomes. The first 20 per task completed in 43.2 minutes; the full 50 completed in another 109.4 minutes under the preregistered runtime-only extension rule. N50 supersedes N20; they are never pooled as 70. All repeated first-20 trajectories matched exactly. No D4/D5 evaluation.

| Distance | Target task | Base SR | Residual SR | ΔSR | N | Rescue / harm | Gain 95% interval |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 31/50 (62%) | 36/50 (72%) | +10 pp | 50 | 12 / 7 | [-6.0,+26.0] pp |
| D1 | `libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 0/50 (0%) | 0/50 (0%) | +0 pp | 50 | 0 / 0 | [-5.8,+5.8] pp |
| D1 | `libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 0/50 (0%) | 0/50 (0%) | +0 pp | 50 | 0 / 0 | [-5.8,+5.8] pp |
| D2 | `libero_goal/put_the_bowl_on_the_stove` | 0/50 (0%) | 0/50 (0%) | +0 pp | 50 | 0 / 0 | [-5.8,+5.8] pp |
| D2 | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 0/50 (0%) | 0/50 (0%) | +0 pp | 50 | 0 / 0 | [-5.8,+5.8] pp |
| D3 | `libero_goal/open_the_top_drawer_and_put_the_bowl_inside` | 0/50 (0%) | 0/50 (0%) | +0 pp | 50 | 0 / 0 | [-5.8,+5.8] pp |

Equal-task-weight gains: G(D0)=+10.0 pp, G(D1)=+0.0 pp, G(D2)=+0.0 pp, G(D3)=+0.0 pp.

Intervals: paired bootstrap where discordance is observed; exact 95% bounds from the zero-discordance probability otherwise. Intervals are per task, not simultaneous across tasks, and do not capture training-seed variance.

No target successes were observed in either policy. These results **do not establish a smoothly decreasing transfer radius**: target base competence is already at the observed floor, so residual generalization and the base's target failure cannot be cleanly separated. No checkpoint, architecture or training change follows these target outcomes.

Read-only inspection of the saved N=20 D1 states finds no target-bowl lift above3cm in either policy on either task. In the ramekin task, minimum gripper-to-target horizontal distance stays≥19.3cm for base and≥18.2cm for residual. A saved-state rendering of plate-task seed4000 shows both policies acting in the empty center. This points to reaching/acquisition failure; thresholds are descriptive, not a validated stage classifier. The full prompt path preserves the target instruction. Source-only alignment losing spatial generalization is plausible, not causally established. No target-driven tuning follows this inspection.

A report-only defect initially rejected the selected intermediate checkpoint because it demanded the full training budget. The aggregator now repeats source-selection verification and enforces the frozen deployment mode. A regression test reproduced the failure before the fix; all 81 tests then passed. Completed rollout artifacts were unchanged.

## 5. Important caveats

Counterfactual diagnostic: eight states from two successful and two failed D0 training trajectories; one candidate action followed by frozen-base continuation. Exact reconstruction and common candidate rollout hashes were verified. This estimates base-continuation outcomes, not unbiased residual-policy Q values.

| Critic | Success-ranking accuracy, non-tied pairs | Harmful edits preferred | Rescue edits rejected | Discounted-return ranking |
|---|---:|---:|---:|---:|
| Auxiliary Cal-QL | 9/16 | 0/4 | 2/2 | 22/62 |
| Fixed-random Cal-QL control | 9/14 | 0/3 | 1/1 | 32/61 |
| Primary after warmup | 10/25 | 2/5 | 2/2 | 29/64 |
| Primary ACTIVE10,095 | 6/25 | 3/5 | 2/2 | 23/62 |

Actor-dependent candidates differ across rows. On identical base/±small actions, discounted ranking is 5/12,6/12,6/12,5/12 respectively: no demonstrated Cal-QL initialization advantage. Mean edit-minus-base Q margin after warmup/10k is−0.00222/−0.00116. Only four independent trajectories make uncertainty large; the 10k discounted-ranking trajectory-bootstrap interval is[0,0.536]. **OTF ranking remains unreliable**, consistent with its poor validation. Candidate count was never increased. Calibration plots, per-state outcomes and paired rescue/harm videos are retained in artifacts.

Remaining fidelity limits: unpublished exact PLD integration/probing range/offline budget; source-only LoRA alignment; chosen SERL SAC LR warmup (DrQ factory differs); PLD AdamW decay unspecified, ours 0.01; raw proprioception, no crop/dropout, independent final Q readouts, Xavier projections, current-state temperature sampling synchronous update ordering and finite-horizon terminal masking. Auxiliary Cal-QL uses gradient clipping unlike its standalone reference. One training seed and selection on D0 validation limit inference; final transfer is exploratory.

Hardware: A100-SXM4-80 GB; Torch 2.7.1+cu128, JAX0.5.3, CUDA12.8; cgroup RAM241.7 GiB. Main run elapsed about 6.49 h including pauses/concurrent validation; learner update compute 3.85 h. Peak Torch allocation 1.97 GiB/reservation 2.27 GiB; whole-device peak 44.21 GiB with one concurrent evaluator, including JAX allocation. Three-process transfer evaluation peaked at 63.16 GiB whole-device memory; this is not standalone learner memory.

Raw commands, hashes, tests, training curves, losses, per-action diagnostics and timing: `outputs/pld_libero/V3-*`. Historical V2 artifacts remain unchanged. Branch `fix/pld-libero-residual-v3`.
