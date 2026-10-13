# TurboVLA progress

2026-10-13: Fresh checkout from exp/turbovla-libero-eval (945a04e), dedicated
branch exp/turbovla-multitask-generalization. GPU4090 is free,24GB.
GitHub account qrvmil authenticated with repo scope.

HF: default cache token invalid; explicit HF_HOME=/workspace/.hf_home token is
valid for qrvmil with repo read/write permissions. Actual private-repo creation,
file upload and authenticated download succeeded in
qrvmil/turbovla-multitask-generalization at commit
973afba1f2c2a25aeb70a295db0b5657cd586894.

Original DINOv3 access currently returns403 GatedRepoError for qrvmil.
User asked asynchronously to enable access to
facebook/dinov3-vitb16-pretrain-lvd1689m. Other preparation can proceed.
No training or new evaluation results exist yet. Previous branch's released
policy results are historical reference only, not this experiment's results.

Ruling: independent A/B/C initialization follows the user's explicit pi0
analogy and the reference REQUEST.md; sequential means execution order.
Ruling: use original pretrained backbones, not released LIBERO-finetuned
DINOv3, to preserve the held-out interpretation.
