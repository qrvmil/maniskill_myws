# RESULTS — PLD LIBERO

## Current headline result
A two-update source-only aligned pi0 checkpoint now exists for engineering validation. No trained LIBERO residual or transfer conclusion exists yet.
The repository was absent and has been cloned. Actual GPU is A100 80 GB, not the requested RTX 5080. All scientific success rates remain PENDING.

## Base policy
Source-only full-model SFT completed two real updates under a 16 GiB allocation ceiling (EXP-000 engineering run). Longer scientific alignment, aligned-base evaluation and successful rollout buffer remain PENDING.

## Source residual training
EXP-002: PENDING. No distillation will be run.

## Cross-task transfer
| Source | Target | Distance | SR base | SR + residual | ΔSR | Episodes | Seed(s) |
|---|---|---|---|---|---|---|---|
| bowl center → plate | source | D0 | PENDING | PENDING | PENDING | 0 | PENDING |
| bowl center → plate | frozen design ladder | D1–D5 | PENDING | PENDING | PENDING | 0 | PENDING |

## Gain vs distance
PENDING. No aggregate from unrun or dummy-policy episodes.

## Negative transfer cases
PENDING. Absence of evaluated cases is not evidence of no negative transfer.

## Runtime / memory
Initial `nvidia-smi`: A100 80GB PCIe, 81920 MiB total, 0 MiB used; driver 595.71.05.
Torch 2.11.0+cu128; CUDA 12.8; CUDA available. This is not RTX 5080 validation.

## Failed runs
- EXP-000 audit: shell `python` absent in noninteractive PATH; use `/venv/main/bin/python`.
- Requested local checkout absent; resolved by cloning public repository.

## Open questions / next experiments
Highest priority: short real source-only full-model SFT with CPU AdamW after successful 12.69 GiB synthetic probe; then aligned-base zero equivalence and source success evaluation. Do not use leaked full-LIBERO checkpoints to bypass this gate.

## Append-only experiment ledger
Entries below are immutable records; corrections are new entries. Summary sections above may be updated.

### EXP-000 / audit / 2026-09-09
Artifacts: `outputs/pld_libero/EXP-000/audit/metadata.json`, task BDDL snapshots.
Exact revision, dirty status and hardware in metadata. Command: repository inspection and `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv`.
Config: `configs/pld_libero/anchor_bowl.json`. Source: spatial bowl from table center to plate. Evaluation tasks: none. Bucket: N/A. Training seed: N/A. Evaluation seeds: none. Checkpoint: none. Episodes: 0. Successes/SR_base/SR_residual/ΔSR: PENDING. Wall time: not instrumented. Peak VRAM: not instrumented; idle usage 0 MiB. No policy executed.

### EXP-000 / full_sft_probe/run-001 / 2026-09-09 19:51 UTC
Actual random-weight full-size pi0 allocation probe, not aligned-base training. Exact command, revision, dirty tree, A100 hardware, torch 2.7.1+cu126 in `outputs/pld_libero/EXP-000/full_sft_probe/run-001/metadata.json`. Failed at forward because synthetic image layout was BHWC; model expects BCHW. Wall 79.692 s; peak allocated 7029305856 bytes, reserved 7096762368 bytes. Episodes 0; task/bucket/seeds/checkpoint/SR/ΔSR N/A. Input-layout bug fixed for run-002.

### EXP-000 / full_sft_probe/run-002 / 2026-09-09
Exact command and metadata: `outputs/pld_libero/EXP-000/full_sft_probe/run-002/metadata.json`; traceback `logs/error.txt`. Random weights, synthetic inputs, full actual pi0, batch 1, checkpointing, bf16, AdamW foreach=False, 16 GiB software allocation ceiling. Actual OOM at optimizer exp_avg_sq allocation requesting 1006 MiB. Forward/backward completed, synthetic loss 2.0881426334381104. Wall 78.029 s. Peak allocated 16075077632 bytes; reserved 16475226112 bytes. Device polling missed the short memory peak; the OOM text reports 15.69 GiB process memory. GPU A100 80GB, torch 2.7.1+cu126. No episodes, no trained checkpoint, no scientific SR/ΔSR. Full SFT with this official optimizer cannot fit the target budget. Official LoRA is the next documented candidate.

### EXP-000 / adapter unit and integration tests / 2026-09-09
`outputs/pld_libero/EXP-000/audit/tests_initial.log`: six expected failures before adapter implementation, one existing SAC test passed. Contract run: seven passed. Real integration initially failed exact RGB equivalence on repeated hard resets (3/6144 channel values differed by 1 in a diagnosed frame); physics/actions matched. Reusing renderer with soft reset fixes full suite: `tests_soft_reset.log`, 9 passed in 8.79s, 3 upstream warnings. Short 8-step fake-model episodes only; not aligned pi0, not a success-rate experiment. Separate source-HDF5 rejection test passed after test-first implementation.

### EXP-000 / smoke-001 / 2026-09-09
Failed before environment creation: CLI named `libero.py` shadowed the upstream namespace package. Renamed to `run_libero.py`. Exact failure and runtime metadata saved in `outputs/pld_libero/EXP-000/smoke-001`. No episodes and no scientific SR.

### EXP-000/smoke-002 / 2026-09-09T19:56:05.553208+00:00
Long dummy-policy rollout failed bytewise RGB trajectory equality. No aligned VLA involved.

Status FAILED; wall 22.159853775054216 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.11.0+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/smoke-002/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/venv/main/bin/python scripts/pld/run_libero.py smoke --output outputs/pld_libero/EXP-000/smoke-002 --episodes 2
```

### EXP-000/smoke-003 / 2026-09-09T19:57:28.513401+00:00
Diagnostic: proprioception and executed actions identical, RGB differed by at most 1 intensity. Failure retained.

Status FAILED; wall 23.059889025986195 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.11.0+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/smoke-003/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/venv/main/bin/python scripts/pld/run_libero.py smoke --output outputs/pld_libero/EXP-000/smoke-003 --episodes 1
```

### EXP-000/smoke-004 / 2026-09-09T19:58:16.411635+00:00
Six 220-step dummy episodes; two zero-residual pairs; 2 Cal-QL + 2 SAC visual updates finite. Cal-QL Q losses 54.98248 and 56.40879; SAC Q losses 1.63489 and 1.41916. These are diagnostics, not base-policy success rates.

Status COMPLETED; wall 62.20914865285158 s; peak allocated 225827840 bytes; peak reserved 230686720 bytes; sampled device peak 1447 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.11.0+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/smoke-004/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/venv/main/bin/python scripts/pld/run_libero.py smoke --output outputs/pld_libero/EXP-000/smoke-004 --episodes 2
```

### EXP-000/smoke-005 / 2026-09-09T20:18:11.908703+00:00
Full 92D physics and actions matched exactly, but RGB differences reached 2 intensity levels and the <=1 gate failed. No tolerance increase accepted silently.

Status FAILED; wall 49.00209615752101 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.11.0+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/smoke-005/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/venv/main/bin/python scripts/pld/run_libero.py smoke --config configs/pld_libero/anchor_bowl.json --output outputs/pld_libero/EXP-000/smoke-005
```

### EXP-000/full_sft_probe/cpu-opt-001 / 2026-09-09T20:02:31.376023+00:00
Synthetic full-model CPU AdamW probe completed. Loss 2.0881426334381104. All parameters eligible for training, no additional freezing. Not an aligned checkpoint.

Status COMPLETED; wall 98.21416974440217 s; peak allocated 13626453504 bytes; peak reserved 13826523136 bytes; sampled device peak 13717 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/full_sft_probe/cpu-opt-001/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/probe_pi0_memory.py --cpu-optimizer --output outputs/pld_libero/EXP-000/full_sft_probe/cpu-opt-001
```

### EXP-000/lora-probe-001 / 2026-09-09T19:59:15.616965+00:00
Official LoRA source-only SFT requested 2 steps; first update OOM. Compiler estimated 15.78 GiB against 14.265 GiB allocation pool. No aligned checkpoint. This fallback was not adopted.

Status FAILED; wall 149.16550545580685 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 15095 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-000/lora-probe-001/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --method lora --steps 2 --output outputs/pld_libero/EXP-000/lora-probe-001
```

### EXP-001/prepare-001 / 2026-09-09T19:55:47.638817+00:00
Converted only 50 source demonstrations, 5882 transitions. Older corpus audit superseded by v2 content-hash-bound conversion; no unseen data.

Status COMPLETED; wall 43.6459541823715 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-001/prepare-001/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py prepare --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --output outputs/pld_libero/EXP-001/prepare-001
```

### EXP-001/prepare-002 / 2026-09-09T20:08:29.870389+00:00
Reconverted the same 50 source demonstrations / 5882 transitions to local/pld_libero_bowl_v2 with corpus content hashes.

Status COMPLETED; wall 42.828392043709755 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 4 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-001/prepare-002/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py prepare --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --output outputs/pld_libero/EXP-001/prepare-002
```

### EXP-001/norm-001 / 2026-09-09T19:57:29.723908+00:00
Official source-only normalization for LoRA feasibility probe.

Status COMPLETED; wall 67.02405156381428 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-001/norm-001/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py norm --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --method lora --output outputs/pld_libero/EXP-001/norm-001
```

### EXP-001/norm-002 / 2026-09-09T20:11:00.770194+00:00
Official source-only normalization for full CPU SFT. Repeated in norm-003 to add producer provenance.

Status COMPLETED; wall 64.76628068834543 s; peak allocated 0 bytes; peak reserved 0 bytes; sampled device peak 742 MiB.
Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree recorded in metadata. Config, complete dirty status and checkpoint: `outputs/pld_libero/EXP-001/norm-002/metadata.json`; losses/episodes/traceback in the same directory. Source is the fixed D0 bowl-center task; no unseen task executed. Training seed 0 where applicable. Scientific SR_base/SR_residual/ΔSR: PENDING.

Exact command:
```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py norm --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --output outputs/pld_libero/EXP-001/norm-002
```

### EXP-000 / guard tests / 2026-09-09
`audit/tests_guards.log`: 14 passed, 1 failed (3/6144 RGB values differed by 1). `audit/tests_guards_002.log`: 15 passed, 3 upstream warnings, 9.81 seconds, with explicit RGB tolerance and exact dummy action/physics digest checks. No policy success claim.

### EXP-000/smoke-006 / 2026-09-09T20:22:57.128625+00:00
Passed six 220-step dummy episodes, two zero-residual pairs and four finite visual RL updates with MSAA disabled. No actual VLA success claim.

Status COMPLETED; wall 64.29305686801672 s; peak allocated 225827840 bytes; reserved 230686720 bytes; sampled device peak 1447 MiB. Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.11.0+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree/config/checkpoint in `outputs/pld_libero/EXP-000/smoke-006/metadata.json`. Source D0 only, no unseen tasks. Scientific episodes 0; successes/SR_base/SR_residual/ΔSR PENDING.

```sh
/venv/main/bin/python scripts/pld/run_libero.py smoke --config configs/pld_libero/anchor_bowl.json --output outputs/pld_libero/EXP-000/smoke-006
```

### EXP-001/norm-003 / 2026-09-09T20:20:47.718292+00:00
Completed source-only normalization with producer provenance binding source corpus and data/model transforms.

Status COMPLETED; wall 67.92912702076137 s; peak allocated 0 bytes; reserved 0 bytes; sampled device peak 726 MiB. Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu126; CUDA 12.6. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree/config/checkpoint in `outputs/pld_libero/EXP-001/norm-003/metadata.json`. Source D0 only, no unseen tasks. Scientific episodes 0; successes/SR_base/SR_residual/ΔSR PENDING.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py norm --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --output outputs/pld_libero/EXP-001/norm-003
```

### EXP-000/full-cpu-sft-001 / 2026-09-09T20:25:42.297968+00:00
Failed strict checkpoint load before any SFT step: meta/to_empty broke tied embedding alias.

Status FAILED; wall 35.21821793727577 s; peak allocated 0 bytes; reserved 0 bytes; sampled device peak 425 MiB. Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree/config/checkpoint in `outputs/pld_libero/EXP-000/full-cpu-sft-001/metadata.json`. Source D0 only, no unseen tasks. Scientific episodes 0; successes/SR_base/SR_residual/ΔSR PENDING.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --steps 2 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-cpu-sft-001
```

### EXP-000/full-cpu-sft-002 / 2026-09-09T20:26:44.759927+00:00
Failed first forward: meta/to_empty left nonpersistent SigLIP position_ids uninitialized, triggering GPU indexing assertion. Discarded meta initialization; official CPU constructor restored.

Status FAILED; wall 62.96692191623151 s; peak allocated 7031698944 bytes; reserved 7096762368 bytes; sampled device peak 7271 MiB. Machine NVIDIA A100 80GB PCIe, 81920 MiB, 595.71.05; torch 2.7.1+cu128; CUDA 12.8. Revision `d2cfebcbd5224bc967fdf8477cec7bf23845d7bf`; dirty tree/config/checkpoint in `outputs/pld_libero/EXP-000/full-cpu-sft-002/metadata.json`. Source D0 only, no unseen tasks. Scientific episodes 0; successes/SR_base/SR_residual/ΔSR PENDING.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --steps 2 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-cpu-sft-002
```

### EXP-000 / checkpoint conversion verification / 2026-09-09
Two official conversions had identical functional tensors but different unused expert lm_head tensors initialized by the upstream constructor. The initial whole-file verification failed as saved in `/workspace/pi0-convert-verify.log`. New conversion at `/workspace/checkpoints/pi0_base_pytorch_v2` has source/checkpoint hashes and exact conversion command in `pretrained_provenance.json`. This remains pretrained pi0, not aligned pi_b. No task episodes.

### EXP-000 / OpenPI normalization test / 2026-09-09
`audit/tests_openpi_contracts.log` first failed a missing test import. Corrected test: `audit/tests_openpi_contracts_002.log`, 2 passed, 14 deselected, 6.48s. Official Normalize/Unnormalize round trip and execution-setting provenance checks passed.

### EXP-000/full-cpu-sft-003 / 2026-09-09T20:28:36.340747+00:00
Actual source-only full-model SFT completed two updates, all 3,501,372,176 parameters eligible, no LoRA or extra freezing. Losses 0.12007024139165878 and 0.12350202351808548; gradient norms 6.074621200561523 and 4.910317420959473; step times 22.6855 and 16.9983 seconds. This tiny checkpoint is for engineering validation only, not a trained scientific baseline.

Wall 237.45498194359243 s; peak allocated 13626460160 bytes (12.69 GiB); reserved 13849591808 bytes; sampled device peak 13739 MiB. GPU A100 80GB; torch 2.7.1+cu128; CUDA12.8. Source D0, training seed0, no environment episodes, no unseen tasks. Checkpoint `outputs/pld_libero/EXP-000/full-cpu-sft-003/checkpoints/2`, bound alignment manifest in parent. Revision/dirty tree, config, exact command and runtime in metadata; per-step losses in logs/sft.jsonl. SR_base/SR_residual/ΔSR PENDING.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v2 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v2 --method full_cpu --steps 2 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-cpu-sft-003
```

### EXP-000/aligned-zero-001 / 2026-09-09T20:33:25.330679+00:00
Aligned checkpoint loaded, but LIBERO import failed because easydict was missing from the separate OpenPI environment. Zero environment episodes completed. Installed missing dependency and added an early simulator-import preflight. Wall 132.45171740837395 seconds; peak allocated 7023468544 bytes; sampled device peak 7187 MiB. Exact command/config/commit/dirty tree in `outputs/pld_libero/EXP-000/aligned-zero-001/metadata.json`; error preserved. Scientific rates PENDING.

### EXP-000/final OpenPI integration tests / 2026-09-09
`audit/tests_final_openpi.log`: 16 passed, 3 upstream warnings, 21.53 seconds in torch 2.7.1+cu128/OpenPI environment. This includes actual LIBERO reset/step, camera/state/action contracts, official model normalization round trip, zero-residual numerical checks, replay, finite Cal-QL/SAC, checkpoint determinism and paired-seed/provenance tests.
