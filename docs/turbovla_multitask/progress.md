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

Latest original-asset download probe: Hugging Face now states that the access
request is awaiting review by the repository authors (still403). The request
has been submitted; authorization has not yet been granted. Do not replace
original DINOv3 with random or LIBERO-finetuned weights while waiting.

User authorized alternate sources, including Meta's GitHub and a public HF
mirror. ViT7B mirror is accessible but architecturally incompatible with the
LIBERO ViT-B recipe (width4096/layers40 versus768/12), so it was not substituted.
Found public original ViT-B/16 weights in nyu-visionx/RAEv2-models, pinned
revision9770b7b980fa1875c8e6d65f226c615c0ce908a8. FullSHA256 matches the published
73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c; the prefix
is pinned in Meta's official hub loader.188 source tensors strict-load in the
Meta model; converted211 tensors strict-load in Transformers. Output cosine
0.9999997616; default max error0.0037375 comes from BF16 saved RoPE periods.
With equal deterministic periods, max error1.6749e-5. This is recorded, not
claimed bit-identical. All learned weights are preserved. assets.py reproduces
this verification and records source and converted-file hashes.

The original-asset access blocker is resolved using the user-authorized public
mirror. assets/manifest.json now exists. Training smoke A completed2 optimizer
updates (first loss0.423989, finite gradient norm1.21527, allocated peak~3954MiB).
Strict reload of all672 tensors and real LIBERO observation→finite(12,7) chunk
passed. Smoke weights remain separate and cannot be published as finals.
Evidence: smoke_audit.json and dinov3_conversion.json.

Supervisor service turbovla_train_a starts the main3001-update A experiment;
poll its actual process and train.jsonl rather than starting duplicate runs.
Report code rejects missing/malformed15-episode cells and recomputes successes
from recorded actions/outcomes.26 scoped tests pass. Remaining scope is still
all three real train runs, verified HF exports,225 rollouts, actual matrix and
executed readable notebook. A main run is not complete merely because smoke
or unit checks pass.

Notebook draft: notebooks/02_turbovla_multitask_eval.ipynb, generated and
nbformat/AST-validated. It exposes model selection, pinned HF download with
hash checks, full strict model load, LIBERO observation,15 seeded rollouts,
episode table/Wilson CI/video and the actual3×5 matrix. It is not yet executed;
publication records and results do not yet exist. build_notebook.py must be
rerun after completed report, followed by full notebook execution and visual
inspection. Expected HF publication metadata location for notebook:
docs/turbovla_multitask/publications/{A,B,C}.json.

Independent read-only code review dispatched per requesting-code-review skill
to agent /root/review_turbovla; findings pending. A training remains live;
latest observed step720,allocated peak4810.9MiB,~0.37s/update. Check actual
supervisor/PID and train.jsonl again on continuation; never infer completion
from this recorded observation. New code and public-mirror evidence pushed
at e5d0755; notebook draft is the next logical code block.

Notebook draft pushed at49521ba. Independent reviewer returned no Critical
issues and one Important wiring defect: publication wrote only runtime metadata
while notebook expected tracked metadata. Fixed publish.py to also write
docs/turbovla_multitask/publications/{variant}.json; fixed local notebook mode
to avoid requiring an HF revision. Reproduced the old local-mode failure from
git history, then exercised the corrected real notebook cell in a CPU test.
27 scoped tests pass. Reviewer found no additional alignment, stats, BERT,
EMA, independence/resume or rollout/action defects; actual success rates and
complete notebook execution remain unverified.

Main A training observed live past1060/3001 updates. Required next steps:
1. Poll supervisor turbovla_train_a plus /workspace/turbovla-multitask/A/train/train.jsonl.
2. Only after actual terminal success and validated final hash, run publish.py
   --variant A and commit/push the generated tracked publication record.
3. Run evaluate.py --variant A under supervisor; collect exactly75 episodes,
   preserve episode-level evidence/video and push a logical evidence block.
4. Start independent B, then C, with the same initialization.json hash; publish
   and evaluate each. Do not restart a live process on observation timeout.
5. report.py builds actual225-episode matrix. build_notebook.py regenerates
   the final tutorial, then execute/inspect its saved outputs and remote weights.

Ruling: automate the already-authorized sequential remaining steps in
run_pipeline.py under supervisor turbovla_experiment. It waits for the live
A service's authoritative terminal state, then verifies final metadata/hashes,
uploads and read-verifies full weights, runs75 episodes, archives and verifies
all episode JSON/video on HF, commits/pushes explicit evidence paths, then
does independent B and C. An OS lock prevents duplicate controllers; the lock
or pipeline_state.json is never treated as proof that a process is live.
Partial evaluation evidence causes a stop for inspection, not replacement.
The pipeline builds the actual matrix only after all three full evaluations;
its terminal phase explicitly leaves final notebook execution/visual QA pending.
Latest observed A step2200/3001; process347730 remains live.27 tests pass and
new pipeline source compiles. Monitor actual supervisor handles and logs.

Pipeline read-only reviewer identified two recovery/publication guard gaps.
Fixed per-variant OS training locks held by train.py itself, plus supervisor
stopasgroup/killasgroup for the controller, so an orphan child cannot race a
replacement. Current A was already running under its separate service and
was not restarted. The waiting controller was deliberately stopped while it
had no training/eval child, to load the fixes safely. Shared exact rollout
contract validation now runs before evaluation upload/commit and final report;
it checks variant, train_tasks, protocol, EMA, precision, seeds and chunk setup.
Observed failing tests for the missing lock/validator, then29 tests pass.
The training math, seed, budget and current A process are unchanged.
