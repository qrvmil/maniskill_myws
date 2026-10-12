# Progress ledger

- Sources pinned: TurboVLA a3365e8f1787d30bafd99d2806edbf4dc82b5370; LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01; reference c6b77f51e3e5bd8873b812d258744c43380fdfeb; HF 9cdb576ed9c56039b9c1149142ed654e876bad76.
- Environment ready, real RTX 4090 CUDA operation verified; torch2.3.1+cu121, MuJoCo3.2.7, robosuite1.4.1.
- Full public checkpoint SHA256 d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab; extracted DINOv3/BERT strictly loaded; complete policy 672 tensors/216073239 parameters.
- Smoke: all five tasks succeed, videos have steps+1 frames. D0 seed10000 reset hash matches reference exactly.
- Policy-only BF16 fresh-process: delta0.9208984375GiB/0.988807168GB, median31.797103ms, 50ms nvidia-smi sampling over init+10warmup+50calls. FP32 1.3876953125GiB/1.490026496GB, median36.566177ms. Both were measured while no simulator/eval processes were active.
- Primary reference n50 run started, foreground exec session26003. D0 completed50/50.
- Independent review found3P2 items; fixed: official task-level RNG stream, standalone pre-benchmark sample preparation, unknown exposure for out-of-training-suite custom tasks. Regression reset(None) test failed then passed. No change to running reference behavior.
- Ruling: do not interpret rollout-memory sampling during primary n50 as isolated policy+sim peak: independent observation-preparation process briefly overlapped it. Preserve those samples labeled concurrent; remeasure isolated one-episode/task rollout after primary exits. Primary episode physics is process-isolated and audited by reset hashes; policy-only memory measurements were earlier and uncontaminated.
- Notebook execution/HTML visual inspection, final250episode/video/reset audit pending.

- Review follow-up: all issues addressed; Wilson 0/50 numerical boundary regression fixed (observed failure then six tests green), charts now clamp interval lengths. Reviewer independently verified all750 preserved pi0 records against source and notebook code parsing.
- D0=50/50,D1=45/50,D2=50/50 verified from authoritative per-task JSON.

- Primary reference run finished successfully: D0 50/50,D1 45/50,D2 50/50,H1 50/50,H2 50/50; all250episodes recorded. Raw sampler with brief concurrent prep is explicitly marked non-isolated and preserved separately.
- Full250video/reset audit and two-episode upstream official action parity now running independently (CPU audit / GPU parity).

- Reference final audit completed: all250reset hashes match preserved pi0 source; all250videos decode and have steps+1 frames.
- Official parity smoke D0 states0/1 matches upstream actions exactly, including episode RNG advancement. Standard n50 run active in exec session99621; CPU results-notebook preflight executed and figures inspected.

- Complete: official n50 D0=50/50,D1=50/50,D2=49/50,H1=50/50,H2=49/50 (99.2% selected-task mean). All250official videos audited.
- Isolated full official rollout memory:1.4541015625GiB/1.561329664GB; CUDA-only policy BF16 repeated0.9208984375GiB/0.988807168GB. GPU PID reporting was empty under real load in this container; added32MiB headless idle guard with red-to-green test. Seven tests pass.
- Both final notebooks execute top-to-bottom (14+8code cells, no errors); saved HTML, figures and Chromium H264 video playback inspected. Reference/official results and source normalization/weights audited in results/completion_audit.json.
- Metadata clarified: official episode key is initial-state index; actual RNG is seed7once per task. No episode actions/outcomes modified. Archive CRC/checksum validation is the final packaging step.
