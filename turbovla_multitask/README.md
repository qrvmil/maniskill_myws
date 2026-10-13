# TurboVLA nested-task generalization

Three independent runs, executed in order, reproduce the pi0 train sets:
A = D0; B = D0+D1; C = D0+D1+D2. H1/H2 are held out from all training,
normalization and checkpoint selection. Task definitions and reference rollout
seeds/horizons come from `configs/pld_libero/multitask_sft.json`.

Initialization must use original DINOv3 ViT-B, BERT and GroundingDINO weights.
**Do not extract the DINOv3 backbone from the released LIBERO policy:** it was
trained on all four suites and would contaminate the held-out comparison.
All runs start from the same seed-0 policy initialization, not the previous run.
The user authorized alternative sources because Meta HF access is pending.
The original Meta ViT-B/16 checkpoint is available from the public
`nyu-visionx/RAEv2-models` mirror; `assets.py` pins its revision, verifies the
fullSHA256 (73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c),
strict-loads it in Meta's original model and Transformers, and checks forward
parity. It preserves all learned tensors. Meta's saved RoPE periods are BF16;
Transformers regenerates FP32 periods. This small deterministic implementation
difference is measured and recorded. No LIBERO policy weights are used.

Data source: pinned `openvla/modified_libero_rlds`, Spatial and Goal suites.
Only the three exact training instructions are extracted into per-task HDF5
files. The pinned official filtered counts are46 D0 /47 D1 /48 D2, rather than
pi0's50 native demonstrations per task. These are the authors' 256px rotated,
replayed no-noop demonstrations;
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
/workspace/turbovla-venv/bin/python turbovla_multitask/assets.py
/workspace/turbovla-venv/bin/python turbovla_multitask/train.py --variant A --smoke-steps 2
# Main runs, executed one at a time; publication verifies remote hashes.
/workspace/turbovla-venv/bin/python turbovla_multitask/train.py --variant A
/workspace/turbovla-venv/bin/python turbovla_multitask/publish.py --variant A
/workspace/turbovla-venv/bin/python turbovla_multitask/evaluate.py --variant A
# Repeat train/publish/evaluate for B, then C. After all225 episodes:
/workspace/turbovla-venv/bin/python turbovla_multitask/report.py
```

HF destination: `qrvmil/turbovla-multitask-generalization` (private).
Work artifacts: `/workspace/turbovla-multitask` (not in git).
TurboVLA revision: `a3365e8f1787d30bafd99d2806edbf4dc82b5370`.
LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

Current status and evidence are in `docs/turbovla_multitask/progress.md`.
The readable tutorial is `notebooks/02_turbovla_multitask_eval.ipynb`.
The full measured matrix is complete: 225 primary episodes across three
independently trained 3001-update policies. The final notebook executes a fresh
15-episode A/D0 run as a reproducibility check and displays the primary matrix.
To regenerate its source, run `python turbovla_multitask/build_notebook.py`;
then `python turbovla_multitask/execute_notebook.py` in the experiment environment.
The latter executes every cell in a fresh registered kernel and exports HTML.
For evaluation the notebook restores DINOv3 and BERT folders directly from
the selected full EMA checkpoint using `prepare_eval_assets.py`; only small
BERT tokenizer files are fetched. Original foundation weights/GroundingDINO
are needed for training initialization, not for this evaluation path.
After final notebook execution and visual inspection, `publish_card.py`
publishes the full verified model card, matrix and executed notebook on HF.

`run_pipeline.py` runs train →verified weight upload →evaluation →verified
evidence archive for A, then B, then C, and pushes each completed block. It
first waits for the already-running `turbovla_train_a` service to become terminal;
it never starts a second copy while that job is live. Run the pipeline under
supervisor (`turbovla_experiment` on this instance). It refuses to overwrite
partial evaluation evidence or to commit unrelated staged changes. Its final
state still leaves notebook execution/visual inspection for the agent.
