/goal

Run a controlled multi-task SFT experiment to test whether the severe specialization
observed after D0-only SFT is caused primarily by SINGLE-TASK fine-tuning.

Work from:

    https://github.com/qrvmil/maniskill_myws
    branch:
        audit/pi0-d0-d1-generalization

Exact starting commit:

    c9b14ef43f7a94213536ebb97f3056bfd9f9dd87

Create a NEW branch:

    exp/pi0-multitask-sft-generalization

Do not modify or overwrite the previous D0/D1 sanity audit results.

After completing the experiment:

    commit all relevant code, reports, figures, notebooks and lightweight results
    push the branch to GitHub

Pushing this new branch is explicitly authorized.


# SCIENTIFIC QUESTION

Previous result:

    D0-only SFT:
        D0 SR rises strongly
        D1 full-task SR remains 0
        partial D1 reach/lift behavior deteriorates
        rollout videos show a strong D0-specific spatial strategy

Main hypothesis:

    narrow single-task SFT encourages a shortcut / overspecialized policy.

A more diverse multi-task SFT dataset may force the model to preserve:

    vision-conditioned behavior
    language-conditioned behavior
    sensitivity to target location

and may improve transfer to tasks not included in SFT.

We want to test whether increasing TASK DIVERSITY during SFT improves generalization.


# IMPORTANT SCIENTIFIC DISTINCTION

Do NOT call performance on a task "generalization" if that task was included in the
corresponding SFT training set.

We need TWO views of the results:

1. TASK PERFORMANCE MATRIX
   Evaluate all trained models on D0, D1 and D2.

   This shows:
       retention
       acquisition
       interference

2. COMMON HELD-OUT GENERALIZATION
   Evaluate every model on the SAME tasks that are excluded from ALL SFT variants.

   This is the clean comparison for:
       "does increasing SFT task diversity improve generalization?"


# EXACT TRAINING TASKS

Use ONE fixed representative task for each nested training level.

D0:

    suite:
        libero_spatial

    task:
        pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate


D1:

    suite:
        libero_spatial

    task:
        pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate


D2:

    suite:
        libero_goal

    task:
        put_the_bowl_on_the_stove


The training sets are nested:

MODEL A — SINGLE TASK

    train = {D0}


MODEL B — TWO TASKS

    train = {D0, D1}


MODEL C — THREE TASKS

    train = {D0, D1, D2}


Every model starts from the SAME official:

    gs://openpi-assets/checkpoints/pi0_base

Do NOT warm-start B from A.
Do NOT warm-start C from B.

All three are independent SFT runs from the same pretrained base.


# COMMON HELD-OUT TASKS

Add at least TWO evaluation tasks that are NOT used for SFT in ANY variant.

HELD-OUT H1 — CLOSE SPATIAL GENERALIZATION

    libero_spatial/
    pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate

This is close to D0/D1:
    same black bowl
    same plate
    same pick/place primitive
    different spatial relation / layout


HELD-OUT H2 — FURTHER PICK-PLACE GENERALIZATION

    libero_object/
    pick_up_the_alphabet_soup_and_place_it_in_the_basket

This changes:
    object
    destination
    language
    scene/layout

but preserves:
    pick/place structure


No H1/H2 demonstration, normalization fitting, checkpoint selection or hyperparameter
choice may use held-out evaluation outcomes.


# OPTIONAL EXTRA HELD-OUT TASK

If runtime is cheap enough, also evaluate:

    libero_goal/
    open_the_top_drawer_and_put_the_bowl_inside

as a more distant compositional held-out task.

But H1 and H2 are mandatory.


# MODELS / TRAINING RECIPE

Use the SAME SFT architecture and optimizer recipe as the prior sanity experiment.

Keep fixed:

    pi0 base
    LoRA rank = 32
    LoRA alpha = 32
    batch size = 8
    training seed = 0
    same image transforms
    same state/action representation
    same OpenPI revision
    same temporal alignment logic
    no EMA
    same optimizer / LR schedule
    same action horizon
    same replan behavior

Primary training budget:

    3001 optimizer updates

for ALL THREE models.

This gives a FIXED-COMPUTE comparison.

Do NOT increase optimization steps merely because there are more tasks.


# TASK-BALANCED SAMPLING

This is critical.

For multi-task SFT, do NOT simply concatenate datasets and let long trajectories /
large tasks dominate.

Sample TRAINING TASKS uniformly.

For model B:

    P(D0) = 0.5
    P(D1) = 0.5

For model C:

    P(D0) = 1/3
    P(D1) = 1/3
    P(D2) = 1/3

Within each task, sample examples normally.

Record actual realized batch/task frequencies and verify they are approximately balanced.


# DATA-VOLUME CONFOUND

The number of unique demonstrations increases as tasks are added.

This means the primary experiment changes BOTH:

    task diversity
    total unique training data

Do not hide this.

Primary experiment:
    use all available 50 demonstrations per task,
    because this is the natural multi-task SFT setting.

Therefore:

    A: 50 demos
    B: 100 demos
    C: 150 demos

Report this explicitly.

If compute/time permits, add a SECONDARY data-matched control:

    fixed total demonstrations ≈ 50

for example:

    A: 50 D0
    B: 25 D0 + 25 D1
    C: approximately 17/17/16 from D0/D1/D2

Use a deterministic preregistered demo subset.

This secondary control is HIGHLY USEFUL because it separates:

    "more tasks"
from
    "simply more unique data"

but do not delay the primary experiment if it materially increases runtime.


# NORMALIZATION

Each SFT variant should compute normalization statistics ONLY from its own training set.

Thus:

    A stats <- D0
    B stats <- D0 + D1
    C stats <- D0 + D1 + D2

This is the valid deployment contract for each trained model.

Do NOT use held-out H1/H2 data to fit statistics.

Record and compare the resulting normalization statistics.

In the report explicitly note that normalization changes with the training corpus
and may contribute to performance differences.


# CHECKPOINTS

Save at:

    0
    500
    1000
    2000
    3001 optimizer updates

for every model.

The main scientific comparison uses:

    final 3001-update checkpoints.

Intermediate checkpoints are diagnostic.

Use exact completed optimizer-update counts in filenames/reporting.

Avoid the old "directory 3000 means 3001 updates" ambiguity.


# FRESH EVALUATION SEEDS

The previous audit used 9000–9049.

Preregister a fresh block before opening any result.

Suggested:

    10000–10049

if there is no collision with existing tracked experiments.

Use:

    N = 50 paired episodes
    per model
    per evaluation task

For the same task, every model must receive the exact same:

    seed
    reset state

Do not change evaluation seeds between models.


# EVALUATION TASKS

Every final model must be evaluated on:

    D0
    D1
    D2
    H1
    H2

Optionally H3 if included.

So the core result is a matrix:

                          EVAL
                  D0    D1    D2    H1    H2

TRAIN D0          ...
TRAIN D0+D1       ...
TRAIN D0+D1+D2    ...


Every cell must also be tagged:

    SEEN
or
    HELD-OUT

For example:

TRAIN D0:
    D0 = seen
    D1 = unseen
    D2 = unseen

TRAIN D0+D1:
    D0 = seen
    D1 = seen
    D2 = unseen

TRAIN D0+D1+D2:
    D0 = seen
    D1 = seen
    D2 = seen

H1/H2:
    held-out for ALL models.


# PRIMARY METRICS

For every model/task:

    successes / N
    SR
    Wilson 95% CI
    mean episode length

For comparisons on identical seeds:

    paired gain
    rescue count
    harm count
    paired bootstrap CI

Do not pool different tasks as independent episodes.


# COMMON HELD-OUT GENERALIZATION METRIC

The cleanest primary generalization comparison should use H1/H2.

Report individually:

    SR(H1)
    SR(H2)

and an equal-task-weight aggregate:

    HeldOutMean =
        (SR(H1) + SR(H2)) / 2

Do NOT weight by number of episodes if N differs.

Primary question:

    Does HeldOutMean increase as SFT task diversity grows?

Do not force monotonicity.


# SEEN/UNSEEN SUMMARY

Also report:

    mean seen-task SR
    mean unseen-task SR

but clearly define which tasks count as seen for each model.

Never average away individual task results.


# BEHAVIOR-STAGE DIAGNOSTICS

For H1 and D1 at minimum, compute the same descriptive diagnostics used in the prior audit:

    Reach < 10 cm
    Grasp contact flag
    Lift > 3 cm
    Task success

Reuse exactly the previous definitions unless there is a correctness bug.

This allows direct comparison with the old D0-only result.

Also calculate these diagnostics on D0 to confirm that multi-task training does not
destroy D0 acquisition.


# VISUAL-SHORTCUT DIAGNOSTIC

Because this experiment is motivated by possible visual shortcutting, include a lightweight
image-sensitivity test on final checkpoints.

For a fixed bank of D0/H1 observations:

keep fixed:

    proprioception
    prompt
    flow noise

compare outputs for:

    correct images
    images shuffled from another scene of the same task

Report:

    first-action L2 change
    first-5-action L2 change
    full-chunk mean L2 change

This is NOT a replacement for the separate attention project.

It is only a behavioral diagnostic.

If the multi-task model is more visually conditioned, shuffled images may perturb its actions more.

Do not assume this outcome.


# REQUIRED MAIN TABLE

Human-readable:

| SFT train set | Eval task | Seen during SFT? | Successes/N | SR | 95% CI |
|---|---|---|---:|---:|---:|

Also produce the compact matrix:

| Train set | D0 | D1 | D2 | H1 | H2 | Held-out mean |
|---|---:|---:|---:|---:|---:|---:|


# REQUIRED FIGURES

Create publication-readable PNG + PDF for every main figure.

At least:

1. GENERALIZATION MATRIX HEATMAP

    rows:
        D0
        D0+D1
        D0+D1+D2

    columns:
        D0
        D1
        D2
        H1
        H2

    values:
        SR %

    visually mark seen vs held-out cells


2. HELD-OUT GENERALIZATION VS NUMBER OF SFT TASKS

    x:
        1, 2, 3 train tasks

    y:
        SR %

    lines:
        H1
        H2
        equal-task held-out mean


3. PER-TASK COMPARISON

Grouped bars or equivalent for D0/D1/D2/H1/H2 across three models.


4. BEHAVIOR STAGES

For H1:

    reach
    grasp
    lift
    success

across:
    D0-only
    D0+D1
    D0+D1+D2


5. SFT TRAINING TRAJECTORY

For each train variant, show relevant task SR at:

    0
    500
    1000
    2000
    3001

Do not overcrowd a single plot.
Split into readable panels/figures if necessary.


6. IMAGE-SENSITIVITY DIAGNOSTIC

Correct vs shuffled-image action change across final models.


# EVALUATION VIDEOS — REQUIRED AND COMPREHENSIVE

Preserve the same standard as the previous sanity experiment.

For EVERY final model and EVERY evaluation task:

    D0
    D1
    D2
    H1
    H2

save:

    first 2 successful rollout videos in seed order, if successes exist
    first 2 failed rollout videos in seed order, if failures exist

If only 1 episode exists in a category:
    save 1.

If 0:
    save none and explicitly record this.

This may produce ~60 videos and that is acceptable.

Each video must:

    show agentview and wrist view side-by-side
    use the same policy observations
    include terminal observation
    display at a useful frame rate
    encode model/train-set/task/seed/outcome in filename

Example:

    train_D0_D1_H1_seed10012_success.mp4


# VIDEO INDEX

Create:

    docs/MULTITASK_SFT_VIDEOS.md

with:

| Train set | Eval task | Seen? | Seed | Outcome | Video | Visible behavior note |

The note must describe ONLY visible behavior, e.g.:

    "moves toward the empty center"
    "reaches the correct bowl but misses grasp"
    "lifts correct bowl and places it on plate"

Do not infer intent from video.

Verify every video:

    exists
    non-empty
    fully decodes
    frame count matches rollout
    seed/outcome matches recorded episode


# HUMAN-READABLE REPORT

Create:

    docs/MULTITASK_SFT_GENERALIZATION_REPORT.md

This report must read like a research collaborator explaining the experiment.

NOT like:

    run logs
    artifact registry
    provenance dump


Required structure:


## TL;DR

5–8 bullets answering:

    Does multi-task SFT improve held-out generalization?
    Is D0 retained?
    Does D1 become solvable once it is trained?
    Does adding D2 help H1/H2?
    Is improvement due only to seen-task exposure or also held-out transfer?
    What happens to reach/lift behavior?
    Does image sensitivity change?


## 1. Question

Why we ran this experiment.


## 2. Experimental design

Clearly show:

    train sets
    seen vs held-out tasks
    data sizes
    optimizer budget


## 3. Main result

Show the full SR matrix immediately.

Explain it in prose.


## 4. Held-out generalization

Focus on H1/H2.

This is the scientifically cleanest section.


## 5. What happens to D0/D1/D2

Separate:
    retention
    acquisition
    transfer


## 6. Behavior stages

Reach/grasp/lift/success.


## 7. Visual sensitivity

Short, careful interpretation.


## 8. Videos / qualitative behavior

Link representative videos:
    D0-only H1 failure
    2-task H1 example
    3-task H1 example
    at least one D0 success per model


## 9. Interpretation

Explicitly answer:

    Is single-task SFT a plausible major cause of the previous specialization?

Distinguish:

    evidence for task diversity
from
    increased total data
from
    normalization differences.


## 10. Limitations

Concise.

Do not dump hashes here.


# NOTEBOOK — REQUIRED

Create:

    notebooks/01_multitask_sft_eval.ipynb

The notebook is intended for a lab seminar AND for quick future re-evaluation.

It must be easy for me to modify:

    checkpoint
    task
    seed list
    N episodes
    video count

from ONE configuration cell near the top.


Suggested notebook structure:

1. What experiment is this?
2. Load config
3. Available trained checkpoints
4. Define evaluation tasks
5. Load one policy
6. Inspect exact prompt / normalization / action contract
7. Run ONE rollout
8. Render / show rollout video inline if practical
9. Run N paired episodes
10. Compute SR + CI
11. Compute reach/grasp/lift
12. Save representative success/failure videos
13. Compare multiple checkpoints
14. Build result table
15. Plot task matrix
16. Example: change task / checkpoint and rerun


The notebook must use reusable functions from the repository.

Do NOT hide all evaluation logic inside notebook cells.

The core evaluation code should live in:

    src/maniskill_myws/...
or a dedicated clean module.


# EASY EVALUATION API

Create a simple reusable entrypoint, for example:

    python scripts/eval_multitask_sft.py \
        --checkpoint <path> \
        --task H1 \
        --episodes 50 \
        --seed-start 10000 \
        --videos 2

or equivalent.

It should be straightforward to evaluate an arbitrary compatible checkpoint/task later.


# REPRODUCIBILITY

Record:

    source commit
    OpenPI revision
    LIBERO revision
    GPU
    CUDA
    Torch/JAX versions
    exact train config
    exact dataset demo lists
    normalization hashes
    checkpoint hashes
    eval seeds

Keep these details in:

    docs/MULTITASK_SFT_METHODS.md
and machine-readable artifacts.

Do NOT clutter the main report.


# TRAINING SANITY CHECKS

Before long training:

1. verify every task dataset is the intended BDDL task
2. verify prompts differ correctly
3. verify action shape/contract = LIBERO 7D OSC
4. verify camera transforms
5. verify task-balanced sampling
6. verify no held-out H1/H2 examples enter training
7. verify train-set-specific normalization
8. verify optimizer step numbering
9. verify LoRA rank32/alpha32
10. verify starting weights identical across A/B/C


# TESTS

Add tests for at least:

    nested train-task definitions
    held-out leakage prevention
    balanced sampling
    normalization provenance
    identical pretrained initialization
    paired eval reset states
    notebook/helper API
    video selection
    video metadata
    action contract
    prompt binding

Run existing relevant tests too.


# HARDWARE / PARALLELISM

This run is on a Vast machine with more VRAM.

Use the extra VRAM for:

    stable training
    parallel evaluation where safe

but DO NOT change scientific hyperparameters merely because memory is available.

In particular:
    keep batch size 8 for the primary comparison.

If evaluating multiple models in parallel changes RNG or numerical behavior:
    run serially instead.


# SECONDARY EXPOSURE-MATCHED ABLATION

If runtime is acceptable after primary results are complete, run one secondary comparison
to address fixed-compute/per-task exposure.

Option A:

    scale optimizer updates approximately with number of tasks:

        D0:          3001
        D0+D1:       6002
        D0+D1+D2:    9003

This roughly equalizes updates-per-task under uniform task sampling.

OR:

Option B:

    data-matched 50-total-demo control

Choose ONE secondary design before observing its results.

Document it clearly as secondary.

Do NOT mix it with the primary table.


# DO NOT

Do NOT:

    use H1/H2 training data
    tune hyperparameters on H1/H2
    use final H1/H2 results for checkpoint selection
    call D1 generalization for the D0+D1 model
    call D2 generalization for the D0+D1+D2 model
    overwrite old audit artifacts
    silently change SFT recipe
    hide negative results


# FINAL DELIVERABLES

At completion I want:

1. final checkpoints for A/B/C
2. full D0/D1/D2/H1/H2 evaluation
3. paired statistics
4. main SR matrix
5. held-out generalization plot
6. behavior-stage diagnostics
7. visual-sensitivity diagnostic
8. comprehensive representative evaluation videos
9. video index
10. human-readable report
11. reproducibility/methods appendix
12. easy reusable evaluation script
13. executable educational notebook
14. tests
15. pushed GitHub branch


# FINAL RESPONSE TO ME

Do NOT answer with artifact paths only.

Summarize in plain language:

    - GPU used / peak VRAM
    - D0-only results
    - D0+D1 results
    - D0+D1+D2 results
    - H1/H2 held-out results
    - held-out mean for 1/2/3 training tasks
    - whether more SFT tasks improved true held-out generalization
    - whether D0 was retained
    - whether partial behavior improved
    - whether image sensitivity increased
    - branch name
    - commit SHA
    - report path
    - notebook path
    - video index path

Then push:

    exp/pi0-multitask-sft-generalization