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

Main A completed3001 updates; final checkpoint1291015370 bytes and SHA
f5fb580ab93d3f66811882b0217eb09878418d66d0c17c1c68ecc5bd81f21154.
Local audit verifies contiguous3001 steps and finite loss/gradients. First100
mean loss0.346116; last100 mean0.089302; allocated peak4810.91MiB.
Initialization SHAea1405220b85277d8caff59b888db27037ec8fc97971da62aa0c160dbeb39976
matches the shared seed-0 initialization. Runtime audit is A/training_audit.json.

Initial HF Xet upload was confirmed live but slow (~0.2–0.24MB/s), at19% after
4.5min. Ruling: enable documented HF_XET_HIGH_PERFORMANCE=1 for upload/download
concurrency, deliberately stop the controller's upload group and resume the
same verified trained artifact using Xet caches. This is a transport optimization
based on measured throughput, not a timeout-based assumption that the process
had died. No training restart. HF model publication remains unverified until
the upload and authenticated read-back/hash checks succeed.

HF_XET_HIGH_PERFORMANCE resolved the slow transfer. Actual A publication and
authenticated download/hash verification completed, then an independent API
model_info check confirmed remote A/final.pth SHA matches local final and size
1291015370. Private repository qrvmil/turbovla-multitask-generalization,
weight revision687d8a127c79a255490f33b27c976019ca7b6cb8. Tracked publication
record publications/A.json. Training evidence and weight record pushed in
e09bd69 by the managed pipeline.

Controller currently running under supervisor turbovla_experiment,PID350369
(old PID349859 was intentionally stopped for transfer optimization). Main
turbovla_train_a is terminal EXITED after success. Current child evaluate.py
--variant A is live; latest observed D0 episode13/15. Preliminary individual
successes are not final task rates; wait for all75 episodes and verify records.
Pipeline will archive/check all JSON+videos on HF, push the completed A eval,
then independently train/publish/evaluate B and C. Do not start duplicate
training or evaluation on continuation. Full notebook execution still pending.

A evaluation fully completed and independently revalidated: exact75 episodes,
15 per task and full reference config/hash contract. Success counts:
D0=14/15(93.33%),D1=0/15,D2=0/15,H1=0/15,H2=0/15.
All episode JSON and videos archived on HF and read-back/hash verified at
9354bd7349bd4228109520876273b7ab7d5b4f70; publications/A_evaluation.json.
Complete A evaluation evidence pushed at c3d4537.

Pipeline now owns live B training childPID355422 under parentPID350369.
Observed step900/3001. B config initialization SHA exactly matches A's
ea1405220b85277d8caff59b888db27037ec8fc97971da62aa0c160dbeb39976,
and its training tasks are exactly D0,D1. This verifies independent original
initialization rather than continuation from A. Revalidate live PIDs on resume.
Current A training/eval requirements are met; B,C,full matrix and executed
notebook remain incomplete. Adjusted matrix styling to the user reference:
same3×5 layout, blue0–100% scale, gold outlines for seen tasks, explicit3001
EMA updates and15 paired resets. Actual rendering awaits all real cells.

Final handoff should also replace the HF repo's initial access-probe status
with a concise root model card linking all three verified model revisions,
normalization files and the runnable notebook; retain the bundled licenses.
Do this from completed evidence, after the pipeline's matrix and notebook QA.

B completed3001 updates. Remote final SHA
6b826614f4c4c0c945767be001b4cfefb58eb73562ace1b4943d4303849f52cf
verified via authenticated model_info at a6684fc322079e39ad98bb9f2211c5839e8b0092.
Contiguous steps and finite losses/gradients verified; first100 mean loss
0.356238, last100 mean0.111333. Initialization matches A exactly.
B full75-episode evaluation revalidated: D0=14/15,D1=13/15,D2=0/15,H1=0/15,H2=0/15.
Weights/training evidence pushed at07de717; complete eval/archive evidence
pushed at711f432. Pipeline has moved to independent C training (observed320
updates, live parent350369; revalidate child PID and progress on continuation).

Evaluation portability improvement: prepare_eval_assets.py restores211 DINO
and199 BERT tensors from our full EMA checkpoint to per-checkpoint-hash folders.
Only BERT tokenizer files are downloaded; no foundation model or GD weights
needed for eval. Actual A restoration strict-loads; every saved encoder tensor
equals its EMA source; cache hashes revalidate. Full CPU policy action output
matches the prior initialization-folder load exactly (max difference0.0).
Evidence eval_assets_audit.json. Notebook regenerated with this clearer flow;
still draft until complete225-episode matrix and full execution/inspection.

Prepared publish_card.py for final HF handoff. It requires all three verified
remote models, matching original init hashes, full matrix and an executed,
error-free notebook before publishing card/matrix/notebook and replacing our
initial access-probe placeholder. Must run after final notebook QA, verify its
downloaded artifacts and push generated metadata; do not execute prematurely.

Current C live childPID360779 under supervisor parent350369, observed2000/3001
updates. C original init SHA equals A and B exactly; train_tasks exactly
D0,D1,D2. This independently verifies all three intended starting points.
Portable notebook changes and actual encoder/full-policy parity audit pushed
at e6d1055. Remaining work: wait for C train/publish/eval/archive and pipeline
report, inspect actual matrix, execute notebook top-to-bottom, inspect saved
images/video/HTML, publish_card.py, push final artifacts and requirement audit.
Parent's terminal models_and_matrix_complete is not full-goal completion;
notebook execution/inspection and final remote artifact checks still remain.


## Completed experiment and notebook verification (2026-10-13)

All three original-initialization runs completed exactly 3001 finite updates.
C loss averaged 0.3520 over its first100 updates and 0.1136 over its last100;
its full remote weight SHA and identical A/B/C initialization were verified.
All225 primary episodes are complete and archived on HF with downloaded-file
hash checks. Successes per15 (D0,D1,D2,H1,H2): A=(14,0,0,0,0),
B=(14,13,0,0,0), C=(14,12,13,0,0). The full CSV independently matches the
validated records; the PNG was inspected visually. Targeted tests:29 passed.

The controller finished evaluation/archive but its final Git whitespace check
rejected CSV CRLF endings. Fixed the writer to emit LF, regenerated only the
report/notebook source, verified git diff --check and published the matrix in
f4d6379. No training or primary evaluation was repeated or overwritten.

Managed notebook PID368970 completed successfully in a fresh kernel: all10
code cells executed, no errors, verified HF download and full policy load,
LIBERO camera preview,15 new A/D0 episodes, summary/table/matrix/video. The
new actions, steps and outcomes match the primary15 A/D0 episodes exactly.
Chromium rendered the HTML offline: all3 tables (3/15/15 rows), both camera
images and full matrix are readable. Embedded512x256 video decodes and plays;
its frame and the notebook introduction were inspected. Execution/render
metadata, screenshots, HTML and episode comparison are saved in this branch.

Final HF card, matrix PNG/CSV and executed notebook were published at
03d6cd674d8b7c9ef184cde698d71490c4784533 and read back with exact hashes.
The model repository remains private, as configured; qrvmil has access.


Final completion audit verifies298 files at one immutable HF snapshot,
03d6cd674d8b7c9ef184cde698d71490c4784533. For each artifact, local SHA256
matches its recorded authenticated-download hash, and remote LFS SHA256 or
Git blob SHA1 matches the exact file content. This includes all3 complete
raw/EMA checkpoints, own statistics, training provenance, all225 primary
rollout records/videos and final card/matrix/executed notebook. Requirement
mapping is in completion_audit.json. All implementation-plan items are done.
