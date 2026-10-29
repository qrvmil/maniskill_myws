# Status — 2026-10-29

Preflight in progress. Long training has NOT started. No new trained checkpoints or evaluation results.

Workspace: `/workspace/visual-alignment-repro-20261029`. Supervisor: `pi0_vram_preflight`. Full allocatable VRAM test runs without a memory cap, tests 0–8 and 10, one pass, 1000 iterations for tests 0/10. The optional three-hour bit-fade test 9 is disabled by the upstream default. The driver/CUDA-reserved portion cannot be tested as a user allocation. No claim of all physical cells or guaranteed future stability.

Machine: requested host150140, instance58231938, machine153731, 1×CMP170HX, NVML65536MiB, CUDA sm80, driver615.71.09, toolkit12.8, x86_64. GPU idle before the test. No persistent volume; remote saves are mandatory.

Local authenticated GitHub and HF identity: qrvmil. App HF connector: dpaleyev, read only. Actual machine credentials are checked independently and never copied into artifacts.
