# Multi-task SFT implementation plan

**Goal:** Execute the user-specified controlled A/B/C study and publish reproducible evidence.
**Architecture:** Add separate protocol/data/training/evaluation/report modules; reuse pinned OpenPI and historical LIBERO transforms, rollout, progress, prompt checks and video validation. Store large assets under /workspace/multitask-sft and lightweight evidence under docs/multitask_sft.
**Spec:** docs/multitask_sft/REQUEST.md (user-supplied executable research protocol).
**Execution:** inline with tests first; final independent review.

## Global constraints
Base c9b14ef43f7a94213536ebb97f3056bfd9f9dd87; no changes to prior evidence.
A={D0}, B={D0,D1}, C={D0,D1,D2}; 50 demos each; independent official pi0_base starts.
LoRA32/alpha32, batch8, seed0, 3001 updates, no EMA; same pinned recipe.
Uniform task sampling; corpus-only normalization; no H1/H2 demo downloads.
Checkpoints use completed updates 0/500/1000/2000/3001.
Final N50 on D0/D1/D2/H1/H2 with paired seeds10000–10049.
Intermediate diagnostics: D0/D1/D2 N50 at all registered milestones; no held-out selection.
Secondary, only after primary completion if affordable: data-matched 50 demos,
B first25 per task, C first17/17/16 in numeric demo order. No H3 planned.

## Review focus
Reject mismatched normalization corpus or base weights; verify real batches are balanced.
Reject incomplete/duplicate/mismatched paired resets; never pool tasks as independent replicates.
Enforce action shape and exact prompt before tokenization; honor arbitrary compatible checkpoint input.
Record absent video categories and decode all selected clips including terminal frame.
Keep initialization perturbation distinct from training, and unseen from universally held-out tasks.

## Tasks
- [ ] 1. Preregister protocol; restore pinned dependencies; write contract tests for nested sets, leakage, sampler, provenance, resets, statistics, videos and prompts.
- [ ] 2. Convert only D0/D1/D2 with historical alignment and camera logic; fit variant-specific official chunk-weighted statistics; bind hashes/demo lists.
- [ ] 3. Wrap official trainer with balanced index sampler and exact saves; verify identical initial state and record realized optimizer-consumed task counts. Train A/B/C serially under supervisor.
- [ ] 4. Reusable evaluation API + CLI; paired resets, progress, videos and fixed same-task image-shuffle bank. Run final full matrix and intermediate diagnostics.
- [ ] 5. Compute per-task Wilson and paired uncertainty plus equal-task held-out mean; PNG/PDF figures; inspect and index all videos; methods and human report.
- [ ] 6. Educational notebook using helpers with one configuration cell; execute/check notebook and relevant existing tests; independent review, commit and authorized push.

## Progress / rulings
2026-09-22: Fresh clone is isolated on requested branch at exact base. User's detailed protocol supplies design and authorization; no redundant approval gate. GPU CMP170HX64GiB; filesystem not a persistent volume. Prior results untouched. Setup wrapper must source Vast utilities before shell strict mode (logging.sh reads optional $1).
