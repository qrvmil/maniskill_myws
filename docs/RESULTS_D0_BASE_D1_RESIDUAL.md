# D0 base / D1 residual adaptation

Status: experiment running. No adaptation or retention conclusion is available yet.

## 1. Hypothesis

A residual trained only on D1 can extend a frozen D0-aligned pi0 to D1 while
preserving D0 competence. D1 performance measures trained task adaptation.
Primary gain is paired D1 delta SR; secondary gain is paired D0 delta SR.

## 2. Exact D0/D1 roles

Both tasks belong to `libero_spatial`.
D0: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
D1: `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
The base sees only D0 demonstrations. Residual replay, warmup and active RL see
only D1. Final evaluation includes exactly D0 and D1.

Preregistered blocks: D0 sanity5000–5049; D1 training6000–6099;
D1 validation7000–7049; final8000–8049 paired within each task.
No existing config collisions were found. Registration and full configuration:
`outputs/pld_libero/V4-preregistration.json`.

## 3. Base D0 result

Reconstructed and permanently frozen before the D1 audit: official `gs://openpi-assets/checkpoints/pi0_base`,
D0-only LoRA32, seed0, batch8, exactly3001 updates. All50 official D0 demos,
5832 correctly shifted observation/action pairs, D0-only normalization.
Frozen checkpoint: `outputs/pld_libero/V4-sft3001/checkpoints/pi0_libero_seen_lora32/EXP-001/3000` (3001 optimizer updates). Alignment manifest SHA256:
`a8811e3426985cd39601a721e8bc8689f145f86fa599d46505c54c37322f0f4f`.
D0 sanity: **32/50 = 64%**. Exact-zero: **50/50 pairs passed** for actions,
physics, images, outcome and length; zero correction also32/50.
Freeze record: `outputs/pld_libero/V4-base-freeze.json`.

## 4. D1 feasibility / offline initialization

Frozen-base D1 audit: **0/50** on the initial registered audit block;
collection completed **0/100** on6000–6099. Every episode lasted220 steps;
no measured grasp or lift>3cm. Mean minimum reach distance was0.256m on the
initial50 (0.242m over100); mean bowl-to-plate progress was−0.0253m
(−0.0200m over100). No successful D1 base replay exists. Collection correctly
closed its Cal-QL gate; its FAILED artifact status records this expected shortfall.
Full evidence: `outputs/pld_libero/V4-feasibility-summary.json`.

Official D1 fallback is being prepared solely for residual replay/auxiliary
full-action Cal-QL. Native demo reexecution failed the strict state/proprio/image
comparison; evidence remains at `/root/pld-runs/d1-demo-verify-native-002/verification.json`.
The pinned official converter drops its initial5 actions and retains original
states while regenerating observations, so its published start does not preserve
the converter's complete controller history. The replacement reconstruction will
save actual continuous action reexecution in the current simulator, with two
independent passes checking determinism and real task success. This is a distinct,
explicit current-runtime observation contract, not a passed native-equivalence test.
Actual frozen base actions on these exact reconstructed observations remain required.
Reachability at xi=.5 will be recorded before RL.

## 5. D1 residual training curve

Pending. Registered checkpoints:5k,10k,25k,50k active steps; conditional100k
only on D1 recovery. The operational criterion, registered before any residual
training in `outputs/pld_libero/V4-stage-preregistration.json`, is strictly higher
D1 deterministic validation SR at50k than25k. Selection requires all four
milestones, plus100k if continued. Episode-boundary overshoot is less than220.
Probe fraction0. Sparse terminal success reward. One seed.

## 6. Selected D1 specialist

Pending. Deterministic actor is preregistered. Select by full50-seed D1 validation
SR, ties smaller correction then earlier checkpoint. Save hashes and paired
validation evidence before opening either final result. A failed improvement
gate will be disclosed explicitly, without tuning on D0.

## 7. Final D0/D1 paired table

Not run. N=50 paired episodes per task is required; no preliminary numbers are
presented as final results.

## 8. Interpretation

No scientific conclusion yet. Positive D1 gain is adaptation; positive D0 gain
is backward transfer, negative D0 gain is interference. Report rescue/harm
counts and paired95% intervals, including negative findings.

## 9. Deviations from PLD

Intentional D0-only LoRA alignment followed by D1-only residual adaptation;
probe fraction0; deterministic deployment fixed from prior V3 evidence.
Any D1 demonstration initialization is an explicit fallback, not base SFT or
residual actor behavior cloning. The auxiliary offline actor is discarded.
V3 implementation caveats in `RESIDUAL_V3_AUDIT.md` remain applicable.
No distillation, second SFT, or base updates from D1.

Measured so far: D0 SFT7009.9s, peak33673MiB; D0 zero audit968.2s, peak9352MiB;
D1 collection1406.8s, peak10090MiB. Peaks are sampled whole-device usage.
Runtime and validation: A100-SXM4-80GB; pinned Torch2.7.1+cu128/JAX0.5.3 and
LIBERO/robosuite/MuJoCo. Full suite88 passed/6 optional skipped before the latest
training integration test, which also passed. Real MuJoCo zero/reset integration
passed, as did pinned SERL feature and distribution/shared-gradient parity.
Pretraining full-suite verification:112 passed,6 optional skipped. Independent
reviews approved staging/export controls and the reconstructed replay contracts;
source and online controller parameters match exactly. Final runtime, peak VRAM and final verification will be recorded here.
