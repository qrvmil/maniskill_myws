# PLD → LIBERO: source specialist recovery

**Status: in progress; D1–D5 closed.** Historical D0 base21/50 vs deterministic residual0/50 is a failed specialist result. No new residual success rate is available yet.

## Setup

D0 remains spatial bowl-center→plate. Train seeds1000–1099; validation2000–2019; final3000–3049, unchanged. No target data or target performance enters selection. No distillation. New branch `fix/pld-libero-residual`, baseline `097f57d`; independent run directories `outputs/pld_libero/V2-*`.

Hardware: A100-SXM4-80GB, RAM limit241.7GiB, initially237GiB free disk. Pinned experiment torch2.7.1+cu128; system torch2.11.0+cu128, CUDA12.8/driver580.159.03. Workspace is ordinary container storage, not a persistent volume.

Source-only alignment is rebuilt from official pretrained pi0 because old binaries are absent. Same source HDF5 SHA `75ede0cf…d3e0b3`, 50 demos/5832 temporally aligned pairs. Official OpenPI JAX LoRA with upstream freeze filter (vision/outer projections also trainable), explicit rank32 in both VLM/action expert, batch8, 4000-update first candidate budget, checkpoints every1000. Selection used all20 D0 validation seeds only: checkpoints after1001/2001/3001/4000 updates scored0/20,9/20,12/20,17/20. Frozen base selected at4000 updates; evidence: `outputs/pld_libero/V2-base-selection.json`.

Frozen-base collection:50 successful trajectories /59 train-seed attempts (84.7%),5604 transitions; replay provenance and exact action=base-action equality verified.

Residual V2: official SERL ImageNet-1K ResNet10 convolutional weights, frozen per-camera trunk and trainable spatial pooling; batch256/replay250k, 100 base-only warmup episodes with actor/alpha frozen, 5000 **active** residual steps for sanity. OTF: one sampled residual plus exact base, min-twin-Q argmax. Deterministic actor and OTF are separately labeled evaluation policies.

## Sanity checks

| Check | Current evidence |
|---|---|
| Baseline integration | 25 passed before changes, real LIBERO reset/step included |
| Updated tests | 54 unit tests + real LIBERO integration + official SERL parity passed; two optional ManiSkill smoke tests not run |
| Encoder initialization | All36 trunk tensors strictly required for all5 networks. JAX/PyTorch comparison passed at32/127/128px, max absolute difference1.08e-4 within mixed tolerance |
| Warmup contract | Unit test verifies exact actor/alpha preservation, critic and target movement; explicit diagnostic-review pause before active interaction; real source warmup pending |
| OTF contract | Seeded shared rollout/eval selector test passes, global RNG preserved |
| Gate1: base/zero | PASS: all20 full-horizon pairs; identical actions, success, length, simulator states and both camera streams, max difference0 |
| Gate2: real warmup/critic | Pending |
| Gate3: short active D0 | Pending |

Source LoRA4000 completed in3014.1s (50.2min), process peak RSS29.5GiB, device peak69987MiB with85% JAX preallocation; this is **reserved process/device memory**, not measured live tensor demand. Checkpoints after1001/2001/3001/4000 updates enter D0 validation. The earlier2-update feasibility probe is not an adequate aligned base.

Batch256 residual feasibility (synthetic batches, not a task result): median Cal-QL0.319s/update, critic-only warmup0.161s, active0.184s; Torch peak2.07GiB, sampled device peak2855MiB. All losses finite. At this measured rate,250k active updates alone would take roughly12.8h plus rollouts/evaluations; no batch/replay reduction is needed for memory.

## Source D0 results

| Checkpoint | Base SR | Deterministic SR | OTF SR | Δdet | ΔOTF | OTF base selection | Mean absolute δ |
|---|---|---|---|---|---|---|---|
| Historical full-SFT3000 / residual50000, final n50 | 42% | 0% | not measured | −42pp | — | — | — |
| V2 source validation n20, base SFT4000 | 85% (17/20) | pending | pending | — | — | — | — |

## Remaining limitations

See [audit](RESIDUAL_FAILURE_AUDIT.md) for verdicts and exact references. No public author PLD training code was located; SERL is a documented reconstruction basis, not claimed exact PLD implementation. Explicit differences: one source specialist/seed, no distillation, fixed base chunk5, base OTF fallback, frozen actor warmup, synchronous1 critic update/environment step, log-alpha surrogate, clipped-density Cal-QL approximation, no random-crop augmentation, separate trainable camera heads for actor/twin critics, finite-horizon masks. Paper does not specify a universal250k interaction budget or exact UTD; 1000 Cal-QL pretrain updates and its conservative coefficient5 are explicit local choices, not verified author settings.

Final seeds were used historically but are held out from all V2 decisions. Main250k residual training and D1–D5 are not authorized by results until all requested source gates pass. Raw logs, provenance, review reproductions and commands remain in artifacts.
