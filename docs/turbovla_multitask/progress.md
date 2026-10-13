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

User explicitly selected original DINOv3+BERT+GroundingDINO initialization.
CUDA torch2.3.1+cu121 tensor operation verified. All five LIBERO environments
reset and render256px; BDDL hashes match reference. Simulator shutdown prints
upstream EGL destructor warnings after successful resets.

Official modified RLDS extraction yielded46/47/48 D0/D1/D2 episodes: the source
filters unsuccessful/no-op demonstrations, so the assumption of50 was wrong.
Ruling: retain all official filtered data with exact pinned counts, report this
difference from pi0's50 raw demos; do not duplicate/fabricate episodes. Initial
preparation intentionally stopped on the50-count assertion; rerun uses the
verified filtered counts and still rejects missing episodes.

Training/evaluation/publication code added; unit checks cover loss masking,
EMA, uniform task sampling, and rejecting partial/smoke weights as final.
Full model training is unverified until original DINOv3 is accessible.

Data preparation finished successfully at15:56 UTC. Supervisor process
turbovla_prepare is EXITED after DATA_PREPARED, not running or stuck.
Verified real frame counts D0=5343,D1=5570,D2=4854; full-file SHA256 checks,
own-corpus stats and finite batches succeeded for all three variants. Image
normalization matches the authors' manual ImageNet normalizer on real frames.
Evidence: data_audit.json. Artifacts: /workspace/turbovla-multitask/data,
A/stats.json,B/stats.json,C/stats.json.

Verification: pytest turbovla_eval/test_eval.py turbovla_multitask -q =>19 passed;
compileall and git diff --check pass. Both train/evaluate CLI --help work.
BERT and GroundingDINO original assets downloaded. Original DINOv3 read probe
still returns403 GatedRepoError. No real training, final checkpoints, evaluation
matrix, notebook results or trained-weight uploads exist yet.

Resume: use HF_HOME=/workspace/.hf_home. First retry assets.py. When original
DINOv3 access succeeds, complete asset manifest, run train.py --variant A
--smoke-steps2, verify full forward/backward/checkpoint reload, then execute
train.py --variant A (3001 updates), publish.py --variant A, evaluate.py
--variant A (15 episodes/task), followed by independent B/C. Use supervisor
for long jobs; shared logging.sh must be sourced BEFORE enabling shell nounset
because it accesses optional positional$1. Continue committing/pushing logical
blocks. Keep full scope active until matrix, executed readable notebook and all
three verified HF weight publications are complete.
