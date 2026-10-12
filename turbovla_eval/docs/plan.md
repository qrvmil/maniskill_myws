# TurboVLA LIBERO evaluation plan

Goal: execute the official released TurboVLA on exactly D0, D1, D2, H1, H2 on this RTX 4090; provide Russian seminar-style notebooks, saved videos, raw results and reproducible VRAM measurements.

Architecture: leave upstream TurboVLA intact; use its strict checkpoint loader and suite-statistics policy. A small local evaluation module exposes named-task selection, seed-generated resets matching the reference experiment and official LIBERO initial-state resets. Notebooks show preparation, inference and rollout code directly.

Requirements: 50 episodes/task, seeds 10000–10049 for reference resets; BDDL hashes and LIBERO revision from reference. Native 256x256 dual-view preprocessing, state dimension 8, action dimension 7, BF16 and 12-step execution from release. Reference D2 horizon is 280 (official Goal suite is 300). No new training. H1/H2 are seen by released TurboVLA. Stream videos to a separate videos directory including terminal frames. No synthetic success or invented measurements.

- [x] Pin sources, install isolated Python 3.10 environment and download complete released checkpoint.
- [x] Resolve backbone initialization without downloading redundant gated weights: extract complete BERT/DINOv3 parameters from publicly released checkpoint if needed; strict-load verify all tensors.
- [x] Implement and check named tasks, BDDL hashes, reset provenance, action shape/finiteness, error handling and video metadata.
- [x] Smoke test all five tasks; then execute 50 episodes each using reference reset protocol, retain per-episode results and video.
- [x] Measure policy-only VRAM in a fresh process, sampling nvidia-smi before launch through initialization/warmup/50 timed calls; also report torch peaks and full rollout memory. Show GB and GiB separately.
- [x] Create and execute tutorial and results/VRAM notebooks top-to-bottom; inspect outputs, validate videos and package deliverables.

Verification: cross-check task bindings with reference source; strict model load; real CUDA forward; 250 unique episode records; success from simulator check_success; video frame counts; verify reference reset hashes where matching environment allows; numerical confidence intervals; clean notebook execution and HTML inspection.
