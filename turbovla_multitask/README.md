# TurboVLA nested-task generalization

Three independent runs, executed in order, reproduce the pi0 train sets:
A = D0; B = D0+D1; C = D0+D1+D2. H1/H2 are held out from all training,
normalization and checkpoint selection. Task definitions and reference rollout
seeds/horizons come from `configs/pld_libero/multitask_sft.json`.

Initialization must use original DINOv3 ViT-B, BERT and GroundingDINO weights.
**Do not extract the DINOv3 backbone from the released LIBERO policy:** it was
trained on all four suites and would contaminate the held-out comparison.
All runs start from the same seed-0 policy initialization, not the previous run.

Data source: pinned `openvla/modified_libero_rlds`, Spatial and Goal suites.
Only the three exact training instructions are extracted into per-task HDF5
files. These are the authors' 256px rotated, replayed no-noop demonstrations;
they are not the 128px native pi0 input files. Use same-step RLDS observation →
action alignment; never apply the native-HDF5 one-step shift to RLDS data.
Compute state mean/std and arm-action min/max separately for A/B/C. Preserve
the binary gripper, pad chunks by repeating the last action and mask padding.
Sample tasks uniformly, then frames uniformly within each task.

Initial controlled budget: 3001 optimizer updates per run, batch 8, LR 5e-5,
300-step linear warmup then constant LR; AdamW beta=(0.9,0.95), eps=1e-8,
weight decay=1e-10, clipping=1.0. BERT stays frozen; DINOv3 and other policy
modules train. EMA decay=0.999; final EMA checkpoint is the predeclared primary
checkpoint. This budget matches pi0's update/batch budget, **not** the paper's
40k updates / batch128 joint training. Report actual convergence and measured
runtime; do not claim to reproduce the paper's success rates.

Primary evaluation: final EMA A/B/C on D0/D1/D2/H1/H2, **15 episodes per cell**,
reference protocol seeds10000..10014, 10 settling steps, 12-action open loop,
reference horizons220/280, action clipping, 256px images. The reference protocol
matches the previous pi0 comparison; it differs from official benchmark initial
states and the official Goal horizon300. Record successes/N and Wilson95% CI.

Required deliverables: full3×5 matrix PNG/CSV, episode-level JSON and videos,
readable notebook showing model load → LIBERO setup → rollout → aggregate,
all three full weights + normalization + initialization/data/config provenance
on Hugging Face. Push each completed logical block to this experiment branch.
Do not mark the experiment complete until all actual training/evaluation and
remote artifact checks pass.

## Environment

```bash
export HF_HOME=/workspace/.hf_home
bash turbovla_multitask/setup.sh
/workspace/turbovla-venv/bin/python turbovla_multitask/prepare.py
```

HF destination: `qrvmil/turbovla-multitask-generalization` (private).
Work artifacts: `/workspace/turbovla-multitask` (not in git).
TurboVLA revision: `a3365e8f1787d30bafd99d2806edbf4dc82b5370`.
LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

Current status and evidence are in `docs/turbovla_multitask/progress.md`.
