# PLD → LIBERO: D0 residual recovery

**Outcome: implementation repaired; working OTF specialist NOT reproduced. Gate3 failed; main250k, final-seed evaluation and D1–D5 were not run.** Deterministic deployment largely recovered toward base performance, while exact training-time OTF remained harmful. Historical42%→0% is preserved, not explained by a uniquely established cause: old binary artifacts were absent.

## Setup

Branch `fix/pld-libero-residual`, baseline `097f57d`; no push. Same D0 spatial bowl-center→plate, train1000–1099, validation2000–2019, final3000–3049; unchanged D1–D5 split. No target data/metrics, no distillation. Every run has a separate artifact directory.

Source-only alignment rebuilt from official pretrained pi0 and the same50 source demos/5832 shifted pairs. Official OpenPI JAX LoRA: rank32 in both VLM/expert, batch8,4000 updates; upstream freeze filter also leaves vision/outer projections trainable. Canonical D0 validation after1001/2001/3001/4000 updates: **1/20,9/20,14/20,18/20**. Selected4000 using only validation. Collection:50 successes/59 train resets,5564 transitions; action=base and provenance verified.

Residual: official SERL ImageNet-1K ResNet10 weights, strict conversion/initialization; frozen convolutional trunk, trainable spatial heads. Batch256/replay250k,1000 Cal-QL updates,100 base-only warmup episodes,1 critic update/step,actor every2,scale0.5. OTF uses exactly1 sampled residual plus exact base, min-twin-Q argmax, no entropy in its target backup. First run freezes actor/alpha during warmup; a single-field ablation enables their updates following literal PLD pseudocode. Both stop after approximately5k **active** steps, preserving complete episodes.

## Sanity checks

| Gate / contract | Evidence |
|---|---|
| Tests | **58 passed,2 optional ManiSkill tests skipped**; real LIBERO integration and official SERL parity included |
| Visual initialization | All36 trunk tensors required in all5 networks; JAX/Torch parity at32/127/128px passed, max absolute error1.08e-4 within mixed tolerance |
| Gate1 base/zero | **PASS:**20/20 pairs identical actions,images,physics,length,success; both18/20. Full base trajectories also match across fresh processes |
| Gate2 warmup | **Mechanics PASS:**100 episodes/12512 steps,base85/100; actor/alpha exactly unchanged,critic changed. On256 successful states MC0.595,Q(base)0.875,Q(random)0.869,Q(mean)0.875; random preference28.5%. Calibration limited:27.3%Q(base) outside[0,1],weak action margins |
| OTF semantics | Shared rollout/eval selector and RNG-isolation tests pass; same candidates,scaling,clipping,critic choice |
| Gate3 | **FAIL for exact OTF:** all completed active checkpoints below base; see paired20-seed results below. Main budget not released |

Additional reproducibility bug: default JAX GPU autotuning changed identical-input/noise base actions across fresh processes (max difference0.020766). Versioned `jax_cuda_autotune0_v1` pins supported XLA settings and JAX/CUDA-plugin versions, rejects conflicts/late configuration. Base candidates,zero,collection,warmup were rerun canonically; all repeated validation base trajectories match. This new JAX issue is not attributed to the historical Torch run.

## Source D0 results

All entries below use the same20 source-validation seeds; **these are not final-test estimates**. Correction is mean absolute executed `a_exec−a_base`; correction and OTF base-selection rates are episode averages.

| Warmup / active steps | Base | Det | OTF | Δdet | ΔOTF | OTF base rate | Mean correction det / OTF |
|---|---|---|---|---|---|---|---|
| Frozen warmup, 1210 | 90% | 60% | 35% | -30pp | -55pp | 64.9% | 0.0111 / 0.0813 |
| Frozen warmup, 3543 | 90% | 95% | 60% | +5pp | -30pp | 68.5% | 0.0077 / 0.0717 |
| Frozen warmup, 5181 | 90% | 80% | 40% | -10pp | -50pp | 66.4% | 0.0083 / 0.0773 |
| Warmup updates, 1174 | 90% | 80% | 45% | -10pp | -45pp | 62.7% | 0.0045 / 0.0857 |
| Warmup updates, 3282 | 90% | 90% | 35% | +0pp | -55pp | 63.8% | 0.0040 / 0.0842 |
| Warmup updates, 5154 | 90% | 85% | 35% | -5pp | -55pp | 60.1% | 0.0051 / 0.0929 |

Warmup-only controls at0 active steps: frozen actor **det0/20,OTF6/20**; updates enabled **det16/20,OTF12/20**; base18/20 throughout. All1000 Cal-QL update metrics and all100 base warmup trajectories match exactly across the ablation. Initial OTF improved, but active learning did not sustain recovery. Active rollout successes:freeze10/29,updates10/28. The deterministic19/20 checkpoint is one extra success on reused selection seeds, not established gain; no specialist was promoted for final evaluation.

Diagnostics support an unresolved mechanism: broad sampled corrections (~0.256 mean absolute magnitude), entropy near its maximum0, and weak critic action discrimination. On256 successful offline states/four draws, raw entropy/Q actor-gradient norm ratios were66–102 (freeze) and6–11 (updates). These are pre-AdamW gradients, not causal proof or measurements on failure states. Warmup updates alone did not fix OTF. Paired failure videos,per-dimension corrections,clipping,Q/MC,losses,entropy,replay fractions and hashes are retained in artifacts.

Hardware: A100-SXM4-80GB; RAM limit241.7GiB;237GiB disk initially free; CUDA12.8,driver580.159.03. Experiment torch2.7.1+cu128; system torch2.11.0+cu128. SFT50.2min,device peak69987MiB with85%JAX preallocation. Residual freeze94.4min / ablation113.7min including review pauses; combined training/eval device peaks46212/45334MiB; Torch allocation peak2.07GiB. Batch256 fits; synthetic active-update timing0.184s implies~12.8h for250k updates alone, but that run was blocked by correctness, not memory. Workspace is not volume-backed.

## Remaining limitations

[Audit](RESIDUAL_FAILURE_AUDIT.md) separates confirmed bugs/mismatches from hypotheses. No public author PLD training code/encoder checkpoint was located: SERL weights are a documented faithful-reference initialization, not claimed exact author weights. Remaining differences/unknowns: one source/seed,no distillation,chunk5,explicit base fallback,frozen-warmup safety variant,synchronous UTD1,log-alpha surrogate,min-twin actor objective (SERL averages),critic-only Cal-QL/fixed random proposal,clipped-action density approximation,no crop augmentation,separate visual heads,finite-horizon masks.1000 Cal-QL updates/coefficient5 and exact offline proposal/UTD are not verified author settings.

**D1–D5 readiness: NO.** Source OTF remains substantially below the reproducible base. More budget is not a validated fix. Next investigation should validate critic action ranking against controlled D0 counterfactual returns and resolve the author offline-critic/temperature recipe; any departure must remain explicit. Final3000–3049 were historically observed but untouched by V2 tuning/evaluation. Raw provenance remains under `outputs/pld_libero/V2-*`; key implementation diff: `V2-audit/provenance/key_changes.patch`.
