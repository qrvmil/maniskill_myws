# GB10 visual-alignment preflight

This is a new experiment, not the historical results elsewhere in this branch.
No new pi0 training, residual learning or evaluation has started.

- Instance: Vast 58229523, one NVIDIA GB10 (SM 12.1), ARM64, Ubuntu 22.04.5.
- Unified memory: 130,594,021,376 host bytes; container limit 125,369,843,712 bytes (116.76 GiB). Host MemAvailable is a live measurement, not a reservation.
- Driver: 615.71.09, advertised CUDA maximum 13.4; system toolkit 11.8.
- Live Vast API pricing: compute $0.3986667/hour; storage $0.0925926/hour; total $0.4912593/hour. User's quoted $0.390/hour is not the observed rate.
- New GitHub branch and new private HF model repository: see `preflight/state.json`. Both passed a nonce write/read comparison using the machine's `qrvmil` CLI credentials. Existing repository visibility was not changed.
- Isolated Python 3.11.17 environment: `/workspace/pi0-gb10-venv`.
- Torch 2.7.1+cu128 / torchvision 0.22.1: actual CUDA attention, interpolation, loss, backward and AdamW weight change passed. This is only a primitive gate, not any of the four requested pi0/residual smoke tests.
- Explicit torchvision 0.22.1+cu128 has no ARM wheel; the official ARM wheel in the same cu128 index is named 0.22.1 and resolves with torch 2.7.1+cu128. No training-method change is implied.
- JAX 0.5.3 installation and CUDA prerequisite test are next. Stop before long runs on any failed compute gate.

## Sources inspected

Three-adaptation source: `48139c7ebba24d180ee2302091232f1b87cd92ec`.
Alignment source: `783bab5dc0502f4c7eccb4d7231a480a88dcc3d9`.
OpenPI: `981483dca0fd9acba698fea00aa6e52d56a66c58`.

The alignment source uses frozen C-RADIOv3-L spatial features, with teacher revision `9d0413465e8a91e67bbf2c1ad342815478d1b906`. Teacher receives the same augmented uint8 RGB as student, divided by 255 with its internal conditioner. The two real cameras' 14x14 grids are bilinearly resized to 16x16; the masked dummy wrist is excluded. Student features come after PaliGemma block 9 (index 8), before the final backbone layers. Its LN+MLP projector is 2048→1024→1024; loss is masked per-patch cosine distance plus flow matching loss. Projector calibration uses a frozen student and is followed by a frozen projector in that repository. Its original training is JAX LoRA on mixed D0+D1; that recipe must not replace this request's full/expert PyTorch budgets or D0-only base.

## Required design constraint before implementation

With frozen visual/language backbone and frozen alignment projector, prefix-only middle-layer alignment has **zero gradient to the action expert**. Merely adding that scalar to expert SFT does not align the action expert. A trainable separate alignment head can receive alignment gradients while the action expert receives flow gradients, but that differs from the source's frozen-projector phase. Any port must explicitly record this distinction, test gradient routing and frozen backbone hashes, and publish the final method before main optimizer runs. No such port is claimed implemented here.

## Intended budgets, conditional on all gates passing

Preserve original: native 50 D0 and 50 D1 demos, D0-only normalization, obs[i]→action[i+1], two cameras rotated 180°, full D0 3000 updates, independent full/expert D1 1500 each, effective batch 8, AdamW (.9,.95), eps 1e-8, wd 1e-10, clip 1, warmup100/cosine 3e-5→3e-6, fixed final checkpoints. Final microbatch requires real memory tests; none chosen yet.

Residual retains frozen D0, earliest successful D1 full teacher among steps250/500/750/1000/1250 on seeds22000–22009; Cal-QL1000; teacher warmup20 episodes; SAC online50000 transitions, UTD1/offline fraction.5/batch256; gamma.99/beta.1/xi2, shaping with terminal potential zero, replay/base-action recalculation, independent teacher/base chunk caches, deterministic final actor. Reset uses set_init_state then10 settle steps. Final evaluation: same seeds30000–30049 per task, horizon220/replan5, 300 adapted-policy episodes plus50 D0-base episodes, paired reset/noise hashes, two-camera terminal-inclusive video, Wilson intervals and paired comparisons. These are targets, not completed results or a finalized preregistration.
