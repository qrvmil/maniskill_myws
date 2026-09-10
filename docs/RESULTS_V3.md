# PLD → LIBERO V3

Status: source-only residual initialization is running; active residual/transfer results are not yet available. No distillation, no target training or target-based selection. Local branch `fix/pld-libero-residual-v3`; no push.

## Changes from V2

[Audit](RESIDUAL_V3_AUDIT.md), [preregistered plan](plans/2026-09-10-pld-libero-v3.md), [configuration](../configs/pld_libero/anchor_bowl_v3.json). Added per-episode base probing without replay insertion; uniform learned Gaussian std; shared actor/critic visual representation with critic-trained heads and actor stop-gradient; mean-Q actor objective; unit-density entropy and separate physical scale; SERL temperature and 2000-step online optimizer warmup; auxiliary full-action Cal-QL actor discarded before online learning. The official SERL pretrained convolution trunk remains frozen, as the actual pretrained reference path specifies. Episode-boundary replay/optimizer/RNG snapshots support exact continuation.

## D0 setup and checks

Primary base **SFT3001** was fixed before V3 residual outcomes. Same D0 source-only LoRA32 weights/normalization as V2; no new SFT. Source/task/train split unchanged. D0 validation expanded to2000–2049; fresh final D0–D3 block4000–4049 registered before target outcomes. Strong SFT4000 remains a separate optional ablation, not an alternative selected by residual outcomes.

| Check | Result |
|---|---|
| Moderate base, D0 validation | **32/50 =64%** (first20 reproduce14/20) |
| Base vs exact-zero | **50/50 exact** actions, success, length, physics and images |
| Separate-process check after reference dependencies | 2/2 exact trajectory/image hashes |
| SERL parity | Official trunk feature parity; Distrax density parity; actual shared Flax graph confirms visual stop-gradient for actor and trainable critic heads |
| Tests | 79 passed; 2 unrelated ManiSkill environment tests skipped (dependency absent). LIBERO and SERL integrations passed. |
| Successful base collection | 50/72 attempts; 5910 transitions |
| Warmup / counterfactual ranking | Cal-QL and warmup in progress |

A100-SXM4-80GB, Torch2.7.1+cu128, JAX0.5.3, CUDA12.8; cgroup RAM241.7GiB. Batch256/replay250k retained. Synthetic batch256 learner peak allocated **1.97GiB** (reserved2.27GiB); timings and process/device measurements are separate in artifacts. Online first stage is **50k ACTIVE** interactions, followed by D0 review for100k/250k, not a5k performance stop.

## D0 checkpoints and selected specialist

Pending. Interim paired D0 validation at5/10/25/50k (first20 seeds); final comparison uses all50. Both deterministic actor and exact one-edit-plus-base OTF are compared. Selection uses D0 only and freezes checkpoint, base and deployment before transfer.

## Frozen D0–D3 transfer

Not run yet. Every task in D1/D2/D3 will be shown individually with paired gain and equal-task-weight bucket means after source selection. No claim about transfer radius is currently supported.

## Caveats

Training probing upper fraction.3, any scale schedule, offline update budget and synchronous update ordering are explicit experimental settings; the exact PLD integration is unpublished. SERL SAC-default LR warmup is a chosen reference setting (DrQ factory defaults differ). Raw proprioception, no visual dropout/crop, independent final Q readouts, Xavier visual projections and current-state temperature sampling remain port differences; online PLD actor:critic1:2 and norm clipping1 are explicit, while auxiliary Cal-QL also uses clipping unlike its standalone reference. One-action counterfactual outcomes use base continuation, so they are diagnostics rather than unbiased residual-policy Q targets. One training seed; small paired gains will be reported with uncertainty.

Raw evidence: `outputs/pld_libero/V3-*`, including command/config/code snapshots, immutable model/data hashes, tests and runtime logs. Historical V2 artifacts are unchanged.
