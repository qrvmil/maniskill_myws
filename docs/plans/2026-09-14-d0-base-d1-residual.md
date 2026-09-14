# D0 base / D1 residual: implementation and preregistration

User specification: attached request dated 2026-09-14. Starting revision
`da51570fd0f16291ef95c8651899884e676bfd67`; local branch
`exp/pld-d0-base-d1-residual`. No remote push.

## Fixed experiment

- D0: `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- D1: `libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- Reconstruct official pi0-base → D0-only LoRA32, seed 0, batch 8, exactly
  3001 optimizer updates. D0 demonstrations alone supply normalization. Freeze
  checkpoint and full provenance before any D1 outcomes. No base selection on D1.
- D0 sanity/zero: 5000–5049. D1 feasibility and training resets: 6000–6099.
  D1 checkpoint validation: 7000–7049. Final within-task pairs: 8000–8049.
- Feasibility: first 50 D1 training resets; attempt up to all 100 training resets
  to collect 50 distinct successful base episodes. If fewer than 50 exist, use
  verified official D1 demos for residual offline replay/auxiliary full-action
  Cal-QL. Record actual successes; never relabel D0 replay or demo actions.
- Offline demo base actions come from the frozen D0 model with its real chunk
  cache, advanced along temporally verified demo observations. Save file hashes,
  trajectory verification, executed actions, base actions, images and MC returns.
- Reachability at xi=.5: absolute required correction, per-dimension quantiles,
  component/action representability and clipping. Diagnostic, no scale tuning.
- Primary SAC unchanged from V3; batch256, replay250k, 50:50 offline/online,
  gamma .99, actor interval2, xi .5, auxiliary Cal-QL1000, LR warmup2000,
  shared pretrained SERL trunk, uniform std, mean actor Q, min TD Q,
  SERL temperature. Probe fraction **0**. Retain V3 100 D1 base-only warmup
  episodes with actor updates, then at least 50k active steps.
- Validate deterministic actor at 5k/10k/25k/50k on all 50 D1 validation seeds.
  Continue to100k only if D1 is learning/recovering at50k. No automatic250k.
- Selection: maximum D1 validation SR; ties smaller mean executed correction,
  then earlier training step. Deterministic deployment fixed in advance.
  Improvement gate is strictly SR(residual)>SR(base); report paired bootstrap
  uncertainty and rescue/harm counts without inventing a later threshold.
- Freeze selected checkpoint, base, deployment and config hashes before either
  final task. If the validation improvement gate fails, report scientific failure;
  any final negative-result evaluation must retain the frozen D1-selected policy
  and explicitly disclose the failed gate. No D0 residual feedback for selection.
- Final: 50 paired base/residual episodes each on D1 and D0. Binary LIBERO success
  unchanged. Save rescue/harm videos; no tuning from final results.
- No distillation, second SFT, D0 residual training, or D1 base updates.

## Execution checklist

- [x] Read required V3 docs, implementation and tests; inspect hardware/repository.
- [ ] Install pinned simulator/OpenPI and run existing V2/V3 baseline tests.
- [ ] Add failing role/provenance/selection regression tests; implement explicit
  alignment, residual-training and evaluation roles in protocol/alignment/runner.
- [ ] Preserve legacy hashes/configs; bind both roles and fresh seeds for V4.
- [ ] Reconstruct and hash fixed D0 base; run D0 sanity/exact-zero equivalence.
- [ ] Run D1 feasibility/progress audit and genuine-success collection.
- [ ] Implement/test verified D1 demo fallback; run reachability audit.
- [ ] Run auxiliary Cal-QL, warmup, active D1 training and D1-only validation.
- [ ] Freeze selection/deployment and run final paired D1/D0 evaluation.
- [ ] Verify artifacts, tests, confidence intervals and videos; write concise
  `docs/RESULTS_D0_BASE_D1_RESIDUAL.md` with runtime/GPU/VRAM and deviations.

Implementation touches the existing protocol, alignment, experiment and selection
modules minimally. Demo replay verification belongs in a separate module. SAC
mathematics and action/chunk contracts remain the V3 implementation. New tests
must reject D1 alignment, D0 replay/training, foreign tasks, changed provenance,
non-D1 selection, and unfrozen final evaluation.


Operational rulings before residual outcomes (2026-09-14):
- Continue50k→100k iff deterministic D1 validation SR50k>SR25k; bind all scheduled candidate evidence to the stage record. Registered artifact `outputs/pld_libero/V4-stage-preregistration.json`. If this conservative rule misses later recovery, the experiment ends at50k without claiming a universal failure.
- Native D1 replay mismatch remains a failed audit. Reconstruct actual current-runtime observation/action transitions by continuous official-action reexecution and independent repeat, keeping only real simulator success. Record the dataset-domain deviation; no threshold weakening or relabelled native observations.
- Previous review agent could not resume (tool thread-limit error); controller implemented the gate fixes and dispatched a fresh independent reviewer. Work remains review-gated before long RL.
