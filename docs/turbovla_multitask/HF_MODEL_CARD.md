---
tags:
- robotics
- vision-language-action
- libero
- turbovla
license: other
license_name: turbovla-and-dinov3
license_link: https://huggingface.co/qrvmil/turbovla-multitask-generalization/blob/main/A/DINOv3_LICENSE.md
---
# TurboVLA: nested-task generalization

Three independently trained policies, executed in order on one RTX4090:
**A: D0; B: D0+D1; C: D0+D1+D2.** All start from identical original
DINOv3 ViT-B/16 +BERT+GroundingDINO initialization. The released all-LIBERO
policy is not used for initialization. H1/H2 demonstrations and statistics
are excluded from every training run.

Each checkpoint contains the complete raw and EMA policy state dicts,
including both backbones. Primary evaluation uses EMA. BERT is frozen;
DINOv3 and remaining policy modules are trained for 3001 AdamW updates
(batch 8,LR 5e-5,seed 0,EMA 0.999).

## Models

| Variant | Training tasks | Verified model revision |
|---|---|---|
| A | D0 | [687d8a127c79](https://huggingface.co/qrvmil/turbovla-multitask-generalization/tree/687d8a127c79a255490f33b27c976019ca7b6cb8/A) |
| B | D0+D1 | [a6684fc32207](https://huggingface.co/qrvmil/turbovla-multitask-generalization/tree/a6684fc322079e39ad98bb9f2211c5839e8b0092/B) |
| C | D0+D1+D2 | [cda0c763a2a9](https://huggingface.co/qrvmil/turbovla-multitask-generalization/tree/cda0c763a2a9a77e7d4d0127e86fb9f732cd0929/C) |

Each model folder includes `final.pth`, its own `stats.json`, exact
training configuration/log, source/initialization hashes and licenses.
`evaluation/` includes 75 recorded episodes and dual-camera videos.

| Task | Suite | Instruction |
|---|---|---|
| D0 | libero_spatial | pick up the black bowl from table center and place it on the plate |
| D1 | libero_spatial | pick up the black bowl next to the plate and place it on the plate |
| D2 | libero_goal | put the bowl on the stove |
| H1 | libero_spatial | pick up the black bowl next to the ramekin and place it on the plate |
| H2 | libero_object | pick up the alphabet soup and place it in the basket |
## Results

Successes out of 15 episodes per cell; seeds 10000–10014 are shared across
models. 10 settling steps, 12-action chunks, 256px images, reference horizons
220 (Spatial) and280 (Goal/Object). Success is the LIBERO simulator predicate.

| Train set | D0 | D1 | D2 | H1 | H2 |
|---|---|---|---|---|---|
| D0 | 14/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| D0+D1 | 14/15 | 13/15 | 0/15 | 0/15 | 0/15 |
| D0+D1+D2 | 14/15 | 12/15 | 13/15 | 0/15 | 0/15 |

![Generalization matrix](generalization_matrix.png)

`generalization_matrix.csv` retains seen/unseen labels, exact rates and
Wilson 95% confidence intervals. With 15 episodes, one success changes the
rate by 6.67 percentage points; these results do not establish universal
zero-shot performance.

## Load and evaluate

The full executed walkthrough is [02_turbovla_multitask_eval.ipynb](02_turbovla_multitask_eval.ipynb).
Use the experiment branch and its pinned environment:
https://github.com/qrvmil/maniskill_myws/tree/exp/turbovla-multitask-generalization

```bash
export HF_HOME=/workspace/.hf_home
bash turbovla_multitask/setup.sh
```

Select A/B/C in the notebook. It downloads model and statistics from the
verified revision, checks SHA256, loads the complete TurboVLAPolicy,
restores both encoders from those full weights, connects LIBERO and
executes 15 seeded episodes. This private repository
requires an authenticated HF token.

## Scope and provenance

Data: pinned `openvla/modified_libero_rlds`; 46/47/48 filtered demonstrations
for D0/D1/D2. Images are already rotated256px no-noop replays; RLDS
observation[t] aligns with action[t]. Statistics are fit to each own corpus.
These data differ from the pi0 comparison’s 50 native 128px demonstrations.
Our 3001-update/batch 8 experiment also differs from the paper’s joint
40k-update/batch 128 recipe; it does not reproduce the paper’s aggregate SR.

Original ViT-B/16 is obtained from a pinned public NYU mirror. Full checksum,
strict Meta/Transformers loads and forward parity are recorded in each
`dinov3_conversion.json`. The small deterministic RoPE rounding difference
between Meta and Transformers is documented there. All three runs use
the same converted initialization.

Upstream TurboVLA: https://github.com/H-EmbodVis/TurboVLA
Paper: https://arxiv.org/abs/2607.27205
Retained DINOv3 license is bundled with each model; consult it and upstream
project licenses for applicable terms.
