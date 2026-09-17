/goal

Run a focused BASE-MODEL sanity audit for the LIBERO setup in:

https://github.com/qrvmil/maniskill_myws

Start from the current branch:

    exp/pld-d0-base-d1-residual

The repository has not changed since the last experiment.

Create a NEW branch, for example:

    audit/pi0-d0-d1-generalization

This task is ONLY about understanding the frozen/base VLA.

DO NOT run residual RL.
DO NOT train a residual policy.
DO NOT modify the previous V4 scientific results.

After the experiment is complete, verified, documented, and all tests pass:

    COMMIT all relevant code + documentation
    PUSH the new branch to GitHub

Do not ask for permission to push at the end; pushing this new branch is explicitly authorized.


# SCIENTIFIC QUESTION

Our previous experiment produced the surprising result:

    D0-only SFT base:
        D0: high success
        D1: 0% success

where:

D0:
    libero_spatial/
    pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate

D1:
    libero_spatial/
    pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate

These tasks are deliberately very close:

    same robot
    same black bowl
    same destination plate
    same pick/place primitive
    only the spatial relation / initial bowl location changes

Before doing more RL work, determine WHY the D0-aligned pi0 base has zero D1 success.

We want to distinguish at least these possibilities:

A. generic pretrained pi0 already has essentially zero LIBERO competence;
B. D0 SFT learns D0 but progressively destroys/generalization to D1;
C. our LIBERO/OpenPI adapter, normalization, prompt, action contract, or evaluation path is broken;
D. official LIBERO-finetuned pi0 succeeds, proving the environment/inference bridge works, while our D0-only model simply over-specializes.


# HARDWARE ASSUMPTION

This experiment should be designed for ONE 48 GB GPU.

Preferred:
    L40S 48 GB
    RTX 6000 Ada 48 GB
    RTX A6000 48 GB

Do NOT assume A100 80 GB.

Previous measured peak for LoRA32 SFT was about 33 GiB, so the existing batch-8 training should fit on 48 GB.

At startup:

    nvidia-smi
    free -h
    df -h
    record GPU model / VRAM / driver
    record Torch/JAX/CUDA versions

Do not silently reduce batch size, LoRA rank, image resolution, or other scientific settings unless an actual OOM occurs.

If an OOM occurs:
    preserve the failure evidence,
    explain the smallest necessary change,
    keep it explicit in the report.


# READ FIRST

Read and understand:

    docs/RESULTS_V3.md
    docs/RESULTS_D0_BASE_D1_RESIDUAL.md
    docs/RESIDUAL_V3_AUDIT.md

    configs/pld_libero/d0_base_d1_residual.json

    src/maniskill_myws/pld/libero_alignment.py
    src/maniskill_myws/pld/libero_backend.py
    src/maniskill_myws/pld/libero_experiment.py
    src/maniskill_myws/pld/libero_protocol.py
    src/maniskill_myws/pld/libero_runtime.py

and the pinned OpenPI implementation / configs used by this repo.

Do not rewrite the pipeline unnecessarily.

Reuse the existing deterministic numerical contract, image transforms,
prompt path, OSC action contract, and paired evaluation machinery.


# SANITY CHECK 0 — VALIDITY OF "RAW PI0_BASE"

Before running anything, answer a subtle but important question:

Can the official pretrained:

    gs://openpi-assets/checkpoints/pi0_base

be scientifically evaluated directly in LIBERO without any LIBERO-specific SFT?

Check carefully:

    action normalization
    state normalization
    embodiment/action dimensions
    policy transforms
    data config
    prompt/image transforms

A model comparison is INVALID if the model cannot produce correctly
de-normalized LIBERO OSC actions under a justified normalization contract.

Therefore distinguish:

1. RAW PRETRAINED pi0_base
2. "step-0 D0 setup":
       pretrained pi0 weights
       + exactly the D0 LIBERO data transforms / normalization
       + ZERO gradient updates

These are not automatically the same scientific object.

If raw pi0_base cannot validly be evaluated because LIBERO normalization
or embodiment statistics are undefined, DO NOT invent normalization.

Instead:

    mark raw pi0_base as "not directly evaluable under a valid action contract"

and use the zero-gradient "D0 setup / step 0" model as the clean pre-SFT baseline.

Explain this prominently in the report.


# SANITY CHECK 1 — PRE-SFT D1 COMPETENCE

Evaluate the valid pre-SFT baseline on BOTH:

    D0
    D1

Use the same inference/action pipeline as all later checkpoints.

Primary object should be:

    step 0:
        official pi0_base weights
        D0-specific LIBERO transforms + normalization
        no optimizer update

If raw pi0_base is also independently valid, evaluate it separately and label it clearly.

Run at least:

    N = 50 fresh episodes per task

Use a fresh seed block that has not been used by V2/V3/V4.

Inspect existing configs/results first and preregister a new block before seeing outcomes.

Suggested if unused:

    9000–9049

for this audit.

Record:

    D0 SR
    D1 SR
    Wilson 95% CI
    mean episode length

and basic behavior-stage diagnostics:

    reaches target bowl?
    grasps?
    lifts > 3 cm?
    successful placement?

Do not use these metrics to alter the model.


# SANITY CHECK 2 — OFFICIAL LIBERO-FINETUNED PI0 POSITIVE CONTROL

Find the ACTUAL official OpenPI checkpoint(s) intended for LIBERO evaluation.

Prefer official Physical Intelligence/OpenPI artifacts.

Potential examples may include pi0 LIBERO or pi0.5 LIBERO checkpoints,
but DO NOT assume checkpoint names or URLs.

Verify from the pinned/current official OpenPI repository/documentation:

    exact checkpoint identifier
    intended LIBERO data config
    normalization
    action horizon
    inference preprocessing

Then evaluate at least ONE official LIBERO-finetuned checkpoint on:

    D0
    D1

using N = 50 fresh episodes per task.

If both official pi0-LIBERO and pi0.5-LIBERO checkpoints are easily available and
compatible, evaluating both is useful, but one valid positive control is sufficient.

This model is NOT part of our scientific training pipeline because it has seen LIBERO data.

It is a POSITIVE CONTROL only.

Purpose:

    if official LIBERO-finetuned model succeeds on D1,
    our simulator / prompts / cameras / action adapter are capable of producing D1 success.

If the official checkpoint unexpectedly gets near-zero success:

    STOP interpreting our D0-only zero-shot result as specialization.

Audit the environment/action bridge first.


# SANITY CHECK 3 — GENERALIZATION THROUGH D0-ONLY SFT

This is the most important experiment.

Train EXACTLY the D0-only LoRA32 configuration used in V4:

    pretrained checkpoint: official pi0_base
    task: D0 only
    50 D0 demonstrations
    LoRA rank 32
    batch size 8
    training seed 0
    same normalization
    same temporal alignment
    same camera transforms
    same action representation
    no D1 data

Save/evaluate checkpoints at approximately:

    step 0
    step 500
    step 1000
    step 2000
    step 3001

IMPORTANT:

The exact meaning of "step N" must be optimizer updates.
Do not repeat the previous indexing confusion where directory 3000 represented update 3001.

In the human-facing report use:

    "0 optimizer updates"
    "500 optimizer updates"
    ...

and separately state the corresponding checkpoint directory names.


# CHECKPOINT EVALUATION

For EVERY checkpoint, evaluate BOTH:

    D0
    D1

using the SAME fixed audit seed block.

N = 50 episodes per task per checkpoint.

Do not change seeds between checkpoints.

This allows paired comparison of how the SAME initial scenes change as SFT progresses.

Record for each checkpoint and task:

    successes / 50
    SR
    Wilson 95% CI
    mean episode length

Also record paired changes relative to step 0 where meaningful.


# PRIMARY GRAPH — THIS IS REQUIRED

Create a clean publication-readable plot:

    x axis:
        D0 SFT optimizer updates

    y axis:
        Success Rate (%)

    two curves:
        D0
        D1

points at:

    0
    500
    1000
    2000
    3001

Include readable confidence intervals / error bars.

The graph should directly answer:

    as the model learns D0,
    what happens to D1 generalization?

Save:

    PNG
    PDF

Use matplotlib.

No ugly debug plot.
Use clear labels, legend, title/subtitle or caption-ready figure metadata.


# SECOND REQUIRED GRAPH — POSITIVE CONTROL COMPARISON

Create a simple comparison figure showing D0 and D1 SR for:

    pre-SFT / step 0
    final D0-only SFT
    official LIBERO-finetuned positive control

If raw pi0_base is separately valid, it can be included as another clearly labeled model.

A grouped bar chart is appropriate.

Save PNG + PDF.


# OPTIONAL BUT USEFUL DIAGNOSTIC GRAPH

If D1 SR falls during SFT, create a behavior-stage graph/table showing at least:

    reach
    grasp
    lift
    final success

versus SFT step on D1.

This helps distinguish:

    "model stops reaching the correct object"

from:

    "model reaches/grips but placement fails".

Only include metrics that can be defined robustly from simulator state.
Document exact definitions.


# PROMPT / LANGUAGE SANITY

Because the main concern is whether D0 SFT causes the policy to ignore the changed spatial instruction,
explicitly verify:

D0 prompt seen by model:
    exact string

D1 prompt seen by model:
    exact string

At each evaluated model, confirm D1 inference is actually using the D1 prompt.

Do NOT merely inspect a config.
Add a test/logging assertion through the real inference path.


# ACTION SANITY

For selected identical D1 initial states, inspect step-0 versus late-SFT base actions.

Save a small diagnostic showing whether SFT drives actions toward a stereotyped D0 pattern.

For example for 10 fixed D1 states:

    compare first predicted chunk / first 5 executed base actions

between:

    step 0
    step 500
    step 1000
    step 2000
    step 3001

Report:

    mean L1/L2 action difference from step 0
    per action dimension if useful

Do NOT mistake normalized OSC command differences for physical meters.


# NORMALIZATION AUDIT

This is mandatory because D1 zero SR may come from D0-only normalization.

Document exactly:

    which normalization statistics are used for step 0
    which for every D0-only checkpoint
    which for the official LIBERO positive control

For D0-only stats:

    inspect where D1 observations/actions fall relative to D0 normalization ranges.

For action dimensions especially report whether D1 expert/demo actions are substantially
outside the D0 training distribution.

Do NOT modify normalization based on the result.

This is diagnostic only.


# IMPORTANT SCIENTIFIC SEPARATION

Do NOT conflate these questions:

1. Does generic pi0 know LIBERO?
2. Does D0 SFT teach D0?
3. Does D0 SFT destroy D1 behavior?
4. Does official full-LIBERO pi0 work in our environment?

The final report must answer each separately.


# EXPECTED INTERPRETATION CASES

Case A:

    step 0 D1 > 0
    later D1 → 0
    while D0 rises

Interpretation:
    evidence consistent with D0 specialization / loss of D1 generalization.

Case B:

    step 0 D1 = 0
    all D0-only checkpoints D1 = 0
    official LIBERO model D1 >> 0

Interpretation:
    generic pi0 + D0 embodiment alignment never had D1 competence;
    zero D1 is not caused by catastrophic forgetting from D0 SFT alone.

Case C:

    official LIBERO model also D1 ≈ 0

Interpretation:
    likely environment / evaluation / checkpoint / action bridge issue;
    investigate before any more RL.

Case D:

    D1 is non-zero at intermediate D0 SFT checkpoints and later collapses

This is especially important:
    report the best D0/D1 tradeoff checkpoint,
    but DO NOT retrospectively redefine it as the scientific base for previous V4 results.


# HUMAN-READABLE REPORT — VERY IMPORTANT

Create:

    docs/PI0_D0_D1_SANITY_REPORT.md

This report MUST be written for a human researcher, not for an artifact verifier.

Write it in a style similar to a careful research collaborator explaining results.

BAD style:

    "artifact hash foo passed; run xyz completed; gate 7 status..."

GOOD style:

    "Before any D0 fine-tuning, the model succeeded on X/50 D1 episodes.
     After 1000 D0 updates, D0 performance increased from ... while D1 ...
     This suggests ..., although ... remains a limitation."

The TOP of the report must contain a concise TL;DR with the actual answer.

Then use this structure:

## TL;DR

3–6 bullets with:
    what we tested
    what happened
    whether D1 was ever non-zero
    whether official LIBERO checkpoint worked
    whether the evidence supports catastrophic specialization
    what this means for the residual-RL experiment


## 1. Experimental question

Explain in plain language why we ran this audit.


## 2. Models compared

Human-readable table:

| Model | Training data | Updates | Purpose |

Clearly distinguish:
    pretrained/step-0
    intermediate D0-only SFT checkpoints
    final D0-only SFT
    official LIBERO positive control


## 3. Main results

Main table:

| Model/checkpoint | D0 SR | D1 SR | D0 successes | D1 successes |

Use percentages AND raw counts.


## 4. SFT trajectory

Embed/link the D0/D1-vs-training-step graph.

Explain the curve in prose.

Do not make the reader infer the conclusion from raw numbers.


## 5. Positive-control result

Explain what the official LIBERO checkpoint tells us about our evaluation stack.


## 6. Behavior analysis

Explain reach / grasp / lift / placement if measured.

Include a compact table/graph.


## 7. Normalization and action-distribution audit

Explain whether D1 lies outside the D0 training distribution.


## 8. Interpretation

Explicitly answer:

    Did D0 SFT destroy previously existing D1 competence?

or:

    Was D1 competence absent from the start?

or:

    Is the current evidence inconclusive?


## 9. Implication for residual RL

Explain what this means for the previous:

    D0 frozen base → D1 residual RL

experiment.

For example:

    If the base never had D1 competence,
    residual RL was being asked to learn a task from essentially zero task prior,
    which is materially harder than correcting a modest base policy.


## 10. Limitations

Short and meaningful.

Do not dump implementation trivia here.


# TABLE AND FIGURE QUALITY

Every table must:

    have descriptive column names
    include raw N
    include percentages where appropriate
    be understandable without opening JSON artifacts

Every figure must:

    have readable axes
    labels
    legend
    concise descriptive title
    no overlapping text
    no debug filenames in visible plot title

Store raw data behind figures as CSV or JSON,
but the reader should NOT need raw files to understand the result.


# TECHNICAL APPENDIX

After the human-readable report, a short appendix may contain:

    exact checkpoint paths
    hashes
    commands
    package versions
    seed blocks
    provenance
    runtime
    peak VRAM

This material must NOT dominate the main report.


# TESTS

Preserve historical V2/V3/V4 behavior.

Add tests for at least:

    correct D0 vs D1 prompts through real inference adapter
    checkpoint/update indexing
    same audit seeds across checkpoints
    correct normalization binding
    official positive-control config isolation
    no D1 data entering D0-only SFT

Run the full relevant test suite.


# DO NOT DO

Do NOT:

    run residual RL
    change D0/D1 task definitions
    train on D1
    tune checkpoint selection using D1
    silently use official full-LIBERO data in our D0-only models
    overwrite prior V4 results
    claim raw pi0_base zero-shot results if the action/normalization contract is invalid
    hide negative or zero results
    stop after implementation without running the evaluations


# FINAL DELIVERABLES

Before finishing:

1. Complete all requested evaluations.
2. Generate all tables and figures.
3. Write the human-readable report.
4. Run tests.
5. Review the report yourself for:
       clarity
       numerical consistency
       figure/table consistency
       no unsupported interpretation.
6. Commit all relevant source/config/report/figure-generation changes.
7. Push the branch:

       audit/pi0-d0-d1-generalization

   to GitHub.

In your final response to me, give a concise HUMAN summary:

    - GPU used
    - step-0 D0/D1 SR
    - D0/D1 SR at 500/1000/2000/3001
    - official LIBERO checkpoint D0/D1 SR
    - whether D1 competence existed before D0 SFT
    - whether D0 SFT degraded it
    - whether our evaluation stack passed the positive control
    - branch name and commit SHA
    - link/path to PI0_D0_D1_SANITY_REPORT.md

Do not return only artifact paths or hashes.
Explain the scientific conclusion in plain language.