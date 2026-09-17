# Evaluation video addendum

The user added this requirement after the official control and primary step-0
evaluation completed and while D0 training was running on 2026-09-16.

- Cover D0 and D1 for every evaluated model, including the auxiliary LoRA
  initialization. Retain the first two successes and first two failures in seed
  order; retain all if a category contains fewer than two episodes. Empty
  categories have no videos. This outcome-based illustrative selection does not
  change evaluation seeds, success rates, model selection or training.
- Use deterministic model/task/seed/outcome filenames and side-by-side agentview
  and wrist views. Videos use the existing rollout images: the same two policy
  cameras, downsampled to 128 pixels per view for recording, at 20 frames/second.
- For the two already-completed model evaluations, replay the selected recorded
  seeds after the existing serial GPU pipeline finishes. Require identical
  seed, outcome, length, reset hash, trajectory hash and image hash. Store replay
  evidence separately and leave original scientific results unchanged.
- Decode each MP4 fully with FFmpeg, count decoded frames with FFprobe, check
  frame count against episode length plus one (including the terminal image),
  and retain seed/outcome and file checksum
  metadata. Repeat verification on the final copied publication assets.
- Write `docs/PI0_D0_D1_SANITY_VIDEOS.md` with model, task, seed, outcome, video
  link and a human-inspected visible-behavior note. Link representative baseline
  D1 failure, final D1 failure, official D1 success and final D0 success from the
  main report wherever those outcomes exist.

CPU media checks may run during SFT. Model replay waits for the main GPU pipeline
to exit, so long GPU stages remain serial. The original fixed D1 seed-9000/9001
recordings are retained as earlier evidence; canonical indexed videos use the
new filenames. No new results are pooled into the 700 original evaluation episodes.
