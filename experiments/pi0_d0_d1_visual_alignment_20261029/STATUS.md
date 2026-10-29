# Status — 2026-10-29

Preflight in progress. Long training has NOT started. No new trained checkpoints or evaluation results.

Workspace: `/workspace/visual-alignment-repro-20261029`. Supervisor: `pi0_vram_preflight`. Full allocatable VRAM test runs without a memory cap, tests 0–8 and 10, one pass, 1000 iterations for tests 0/10. The optional three-hour bit-fade test 9 is disabled by the upstream default. The driver/CUDA-reserved portion cannot be tested as a user allocation. No claim of all physical cells or guaranteed future stability.

Machine: requested host150140, instance58231938, machine153731, 1×CMP170HX, NVML65536MiB, CUDA sm80, driver615.71.09, toolkit12.8, x86_64. GPU idle before the test. No persistent volume; remote saves are mandatory.

Local authenticated GitHub and HF identity: qrvmil. App HF connector: dpaleyev, read only. Actual machine credentials are checked independently and never copied into artifacts.

Update: protocol approved — alignment of the common D0 base, original expert/residual frozen scopes. CPU data audit passed:50/50demos,5832D0+5913D1aligned frames; recomputed D0 normalization exact SHA256a06c2e745bf91d8dd6a3fa6377560d3104e6cad621da4ecebe1372c74bbb388e. Original12contract/statistical tests and2JAX/PyTorch alignment-port tests passed;7fail-closed VRAM-gate tests passed.

Official base fetched as33immutable GCS generations; all MD5 checked. Official pinned fp32 conversion matches historical initial weights exactly:8df3cbff6d06f55fc77202b7d66c671c5a9e05403620bd05c8eac95955f09926. Model smoke will use this fresh official base; historical D0 weights remain an inspected reference only.

PyTorch2.7.1+cu128/JAX0.5.3 and pinned Transformers4.53.2 plus the pinned OpenPI transformer replacements are installed. GPU framework/model probes await successful uncapped VRAM test. Model preflight service is installed but STOPPED; long training is not armed.
