# TurboVLA experiment implementation plan

Goal: Train and evaluate the full nested-task experiment requested in the
pasted user instruction, preserving all three policies on Hugging Face.

Spec: `turbovla_multitask/README.md`. Execute inline in the dedicated fresh
checkout and experiment branch. The user already authorized execution and
GitHub/HF publication; no additional publication confirmation is needed.

- [x] Inspect pi0 and TurboVLA reference code and freeze task/protocol definitions.
- [x] Verify HF authenticated read/write, GitHub authentication and free4090.
- [x] Set up pinned Python3.10 / torch2.3.1 / LIBERO / TurboVLA environment.
- [x] Test chunk boundaries, normalization round trip and held-out exclusion.
- [x] Extract only D0/D1/D2 from pinned official modified RLDS suites.
- [x] Verify pinned official filtered46/47/48 demonstration counts, image/state/action dimensions and hashes.
- [x] Download original DINOv3/BERT/GroundingDINO; strict-check initialization.
- [x] Implement full train/checkpoint/resume/EMA/provenance paths and smoke one update.
- [ ] Train A, publish weights, evaluate15 episodes/task; commit/push evidence.
- [ ] Train B independently, publish/evaluate; commit/push evidence.
- [ ] Train C independently, publish/evaluate; commit/push evidence.
- [ ] Generate and verify matrix, report and step-by-step evaluation notebook.
- [ ] Verify every remote weight/stat/config file and complete all requirements.

Review focus: no all-LIBERO policy initialization; no held-out stats; no RLDS
one-step shift; no padded-loss contribution; no evaluation of stale checkpoints;
no synthetic/pending values presented as measured successes.
