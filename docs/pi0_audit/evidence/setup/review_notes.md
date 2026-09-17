# Independent review notes

First code review found no major scientific/numerical blocker in training settings,
checkpoint step mapping, primary0/LoRA initialization distinction, official-control
statistics, prompt path or common rollouts. It identified missing audit-entrypoint
source snapshots, since historical RunArtifacts includes only *libero*.py scripts.
Audit-only source snapshotting was added before training or D0 model evaluation.
The already-running control's later source capture is explicitly labeled, not
claimed to be an exact launch snapshot. No historical logger was changed.

Second reporting review found two interval edge cases: comparing baseline to itself
should give exact zero, and all-gain/all-loss paired samples should not report
zero-width bootstrap uncertainty. Regression tests reproduced both. Identity
comparisons now return [0,0]; constant boundary samples use an exact binomial-mass
bound allowing worst-case opposite-sign discordance. The revised suite passes.

Video-feature review found that pre-action frames alone omit the final outcome.
A regression test failed with 3 frames instead of 4; the encoder now appends the
terminal next_images and verifies episode length + 1 frames. CPU tests also check
that the decoded terminal image contains the terminal pixels, outcome selection,
replay hash/seed/outcome rejection, full decoding and wrong frame-count rejection.
All three video tests pass. Original baseline/control result rows remain unchanged;
video backfill uses isolated replay evidence and strict original-trajectory checks.

## Report identity and freshness review

Independent reviewer found two reporting gaps: checkpoint labels were not cross-checked against saved optimizer-update records, and the report writer could combine fresh tables with preexisting figures/action CSVs. Added strict model/path/seed/action-contract/update-record validation with a red-to-green regression test. The writer now regenerates derived tables/figures and revalidates all publication videos before writing prose. Follow-up read-only review found no remaining actionable issues in these fixes.

## Relocated evidence verification

A follow-up reproduction check found that strict identity validation assumed the evidence remained at its original filesystem path. Added a regression using relocated metadata without weights; it failed before the fix and passes afterwards. Validation now anchors original checkpoint paths to the registered zero-update record, preserving milestone/path checks while allowing the archived evidence to regenerate figures. Verified all five currently archived model identities from the repository evidence path.

## Completed numerical and figure review

Independent read-only CPU review checked all700 originalepisodes,14summaries, Wilsonintervals, pairedreset hashes, stagecounts, episodetimes/lengths, actiondifferences/clipping and recordedsource/artifacthashes. No actionable numerical or figure discrepancies were found. All three figures accurately display the measured results. The relocated-evidence identity validation and regression were also reviewed and accepted.

## Final human report and video review

Independent read-only review found no remaining actionable issues in the human report. The interpretation, tables and figures agree with original results. All 38 videos independently passed full decode, frame-count, checksum, identity, replay and outcome-coverage checks. Sampled key clips support their visible-behavior descriptions. Corrected ambiguous wording to state that the real MuJoCo integration test passed; four optional tests were skipped. All report links resolve and historical scientific files remain unchanged.
