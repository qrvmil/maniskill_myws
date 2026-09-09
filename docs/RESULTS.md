# RESULTS — PLD LIBERO

## Current headline result
LIBERO integration and corrected source-only full-model pi0 SFT smoke runs work. Both corrected CPU-SFT zero-residual pairs matched exactly; the tiny base scored0/2 on source-validation seeds. Official GPU SFT completed2 finite updates on the user-authorized A10080GB, with29.99GiB peak PyTorch allocation. GPU-checkpoint zero validation and the100-step alignment pilot are next. **No learned residual or cross-task transfer conclusion exists.**

## Base policy
Corrected corpus:50 source demos,5832 observation/action pairs. All earlier same-index checkpoints are excluded from scientific use. Completed2-update checkpoints are engineering fixtures; adequate source alignment and successful frozen-base offline collection remain PENDING. User authorized A100 usage; current experiments no longer impose16GiB.

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
Highest priority: validate official GPU SFT on the user-authorized A100, run the registered100-step source-only pilot, then source-validation success and successful-base collection. Do not use leaked full-LIBERO checkpoints to bypass this gate.

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

### EXP-000/aligned-zero-002 / 2026-09-09T20:37:39.699382+00:00
Completed four real pi0 episodes (two base/zero pairs), seeds2000,2001, D0 source task ID2. All 220 steps per episode. Base successes0/2; zero-residual successes0/2; both SR0, ΔSR0. **This tests an exact zero correction, not a learned residual and not cross-task transfer.** Every RGB/action/full92D-physics maximum absolute pair difference was0. Source SFT initial-state overlap check passed.

Checkpoint `outputs/pld_libero/EXP-000/full-cpu-sft-003/checkpoints/2` (metadata checkpoint null is a logging omission corrected in code; alignment manifest identifies the actual checkpoint). Wall 227.47734304331243 s; allocated 7162114048 bytes (6.67 GiB); reserved 7275020288 bytes; sampled device peak 8179 MiB; peak process RAM 16360914944 bytes. GPUA100, torch2.7.1+cu128. Exact command/config/revision/dirty status in metadata, base-inference timings in logs, per-episode states and pair checks in eval. This metadata episodes=2 counts pairs; actual environment episodes=4.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-000/full-cpu-sft-003/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-000/aligned-zero-002
```

### EXP-001/sft-pilot-100 / RUNNING
Registered before source success was observed. Supervisor `pld_libero_sft` resumed the two-update checkpoint toward100 total source-only updates; 16CPUthreads, ordinary AdamW unchanged. Final checkpoint/loss/runtime/SR: PENDING. Concrete configuration/command and machine data in `outputs/pld_libero/EXP-001/sft-pilot-100/metadata.json` once initialized.

### EXP-000 / CPU optimizer and summary tests / 2026-09-09
`audit/cpu_adam_threads.json`: representative 67,108,864-element bf16 CPU AdamW updates averaged 0.1253/0.0747/0.0569/0.0552 seconds at 4/8/16/32 threads. This microbenchmark does not predict full training throughput. Fused CPU AdamW was faster but differed in49/16384 bf16 parameter entries after10 test updates (max7.6294e-6); it was not adopted. The optional resident CPU parameter path passed exact small-model comparisons. `audit/tests_summary_resident.log`:17 passed,1 skipped,7.57s, including negative-gain preservation and duplicate-run rejection. These are synthetic unit fixtures, not experimental transfer numbers.

### EXP-001/sft-pilot-100 / INTERRUPTED — temporal-alignment correction
Stopped the supervisor job after46 completed updates (44 additional updates after the two-update resume). No100-update checkpoint exists. `metadata_before_stop.json` preserves the original running metadata; `metadata.json` records the explicit interruption, with per-step peak allocation13626460160 bytes and peak reserved13849591808 bytes from actual logs. Device-wide polling history was held in memory and was lost on supervisor termination; no final sampled peak is invented.

The native upstream `scripts/create_dataset.py` calls env.step(action) before storing the observation at the same index. A source-only simulator check at demonstration0 indices10/30/50/70 found stored end-effector positions 0.0061–0.0111m from same-index states but only0.00027–0.00045m from next-index states (`EXP-000/audit/demo_timing_check.json`). This supports the observed one-step offset. The naive native conversion is superseded; all prior SFT/zero runs remain real engineering results but cannot be used as the scientific aligned base. No unseen rollout caused this change.

### EXP-001/prepare-003 / 2026-09-09T20:55:47.564393+00:00
Corrected native post-action timing:50 source demos,5832 training pairs from5882 raw frames. Dataset local/pld_libero_bowl_v3. Its first-observed-state holdout fingerprints were added from the same SHA-verified source HDF5 before normalization; original audit preserved as source_audit_before_holdout_amendment.json. Metadata, dependency versions and code snapshot in the run directory. Wall 42.479561476036906 seconds. No GPU policy episodes; SR/ΔSR N/A.

### EXP-001/norm-004 / 2026-09-09T20:59:37.001663+00:00
Official normalization completed on all5832 corrected source-only pairs. Producer manifest binds corrected corpus/data transforms/statistics. Wall 63.48172018863261 seconds; no policy episodes. Exact command/config/commit/dirty state and package versions in run metadata/artifacts.

### EXP-000/safe fast loading / 2026-09-09
Transformers no_init_weights + official CPU model constructor + explicit restored weight tying passed strict pretrained loading in3.509s. All three registered buffer groups validated, including exact arange vision position IDs; `audit/no_init_load.json`. This avoids the failed meta/to_empty route. Full-model GPU training/inference tests remain separate. `audit/tests_timing_v3.log`:18 passed,1 skipped,7.55s including corrected temporal pairing.

### EXP-000/full-cpu-sft-v3-resident-001 / 2026-09-09T21:01:29.422117+00:00
Corrected native temporal alignment, same50 source demos/5832 pairs, full-model CPU AdamW with one GPU model and resident CPU optimizer parameters. Both actual updates finite: losses 0.1739734560251236 / 0.1942126452922821; gradient norms 7.129666805267334 / 6.614439964294434. Step times 12.990240005776286 / 12.108901647850871 seconds. All 3501372176 parameters remain eligible. No environment episodes and no scientific SR/ΔSR yet.

Status COMPLETED; wall 148.16044089384377s; allocated 13626460160 bytes; reserved 13849591808 bytes; sampled device peak 13739MiB; process peak RAM 30756507648 bytes. GPUA100, torch2.7.1+cu128, CUDA12.8, training seed0, sourceD0 only. Checkpoint `outputs/pld_libero/EXP-000/full-cpu-sft-v3-resident-001/checkpoints/2`; source/corpus/norm/checkpoint provenance in alignment_manifest.json. Revision `a6274bdabf024ae270689000c39d587d066de311`, dirty tree and code snapshot recorded in run directory.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_cpu --steps 2 --optimizer-storage resident_cpu --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-cpu-sft-v3-resident-001
```

### EXP-000 / corrected integration tests / 2026-09-09
`audit/tests_corrected_full.log`:19 passed,3 upstream warnings,12.97s. Command `PLD_LIBERO_INTEGRATION=1 scripts/pld/libero_python.sh -m pytest -q tests/test_pld_libero.py`. Includes actual LIBERO reset/step plus action/camera/proprioception/chunk/zero/replay/finite Cal-QL and SAC/checkpoint/paired-seed/provenance/temporal-conversion/negative-summary tests. Test fixtures do not establish learned-policy transfer.

### EXP-000/full-cpu-sft-v3-move-001 / 2026-09-09T21:07:51.711806+00:00
Corrected full-model two-update reference with whole-model CPU movement. Losses and gradient norms exactly match resident_cpu; all777 saved tensors match bitwise (`audit/storage_full_model_comparison.json`, comparison1.8596s). Measured update times17.6977/14.6454s; wall162.2897066436708s; allocated13626460160 bytes, reserved13849591808 bytes; sampled device peak14461MiB. The integration test overlapped the checkpoint-save phase and added a renderer context, so this device-wide peak is not an isolated training-only measurement. The resident run's isolated device peak remains13739MiB. CPU RAM peak30674673664 bytes. GPUA100, torch2.7.1+cu128, CUDA12.8, sourceD0/training seed0/no episodes. Checkpoint `outputs/pld_libero/EXP-000/full-cpu-sft-v3-move-001/checkpoints/2`; dirty revision `a6274bdabf024ae270689000c39d587d066de311` and code/config snapshots in run. No scientific success/transfer claim.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_cpu --steps 2 --optimizer-storage move_model --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-cpu-sft-v3-move-001
```

### EXP-000/aligned-zero-v3-001 / 2026-09-09T21:11:27.368971+00:00
Corrected-data two-update checkpoint passed both full220-step zero-residual pairs, source-validation seeds2000/2001. All RGB/action/full92D-physics max differences exactly0. Base0/2 successes; zero-residual0/2; bothSR0, ΔSR0. This is an engineering zero-correction test, **not learned residual gain**. Source holdout checks include native first and successor observed states. Four actual environment episodes (metadata episodes2 counts pairs).

Wall160.51728006079793s; allocated7162114048 bytes, reserved7275020288 bytes; sampled device peak8179MiB; CPU RAM15378632704 bytes. GPUA100, torch2.7.1+cu128, CUDA12.8; checkpoint `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-000/full-cpu-sft-v3-resident-001/checkpoints/2`; dirty revision `a6274bdabf024ae270689000c39d587d066de311` and exact configuration/code snapshots in run directory.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-000/full-cpu-sft-v3-resident-001/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-000/aligned-zero-v3-001
```

### Hardware scope changed by user / 2026-09-09
User explicitly instructed us to use the actual A100. Current configurations permit80GiB and upcoming alignment uses pinned official OpenPI PyTorch GPU AdamW. Previous16GiB probes/CPU offload remain historical results; they do not establish RTX5080 compatibility. The scientific split and leakage rules are unchanged. The prepared corrected CPU100-step job was not started; GPU2-step probe then100-step pilot supersede it. No unseen results informed this change.

### EXP-001/norm-005 / 2026-09-09T21:16:20.836347+00:00
Source-only normalization for official full_torch config, same corrected5832 pairs.

StatusCOMPLETED; wall63.26867847144604s; allocated0 bytes, reserved0 bytes; sampled device peak4MiB; CPU RAM peak1372565504 bytes. A10080GB, torch2.7.1+cu128, CUDA12.8. SourceD0/training seed0; no environment episodes, SR/ΔSR N/A. Checkpoint `None`. Revision `a6274bdabf024ae270689000c39d587d066de311`, dirty tree, source/data/norm provenance and code/config snapshots in run directory. GPU smoke snapshot predates optimizer-retention support; it has one final checkpoint only.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py norm --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_torch --output outputs/pld_libero/EXP-001/norm-005
```

### EXP-000/full-gpu-sft-v3-001 / 2026-09-09T21:17:40.001026+00:00
Official OpenPI GPU train_loop completed2 full-model updates, losses0.1740/0.1942 and gradient norms7.13/6.61 (native logs rounded). Measured intervals1.9/0.8s include data/compute but exclude final save. All3501372176 parameters eligible. Checkpoint2 means2 completed updates; upstream final-save counter corrected by wrapper.

StatusCOMPLETED; wall204.38602709770203s; allocated32200932352 bytes, reserved32621199360 bytes; sampled device peak31641MiB; CPU RAM peak16233443328 bytes. A10080GB, torch2.7.1+cu128, CUDA12.8. SourceD0/training seed0; no environment episodes, SR/ΔSR N/A. Checkpoint `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-000/full-gpu-sft-v3-001/checkpoints/pi0_libero_seen_full_torch/EXP-001/2`. Revision `a6274bdabf024ae270689000c39d587d066de311`, dirty tree, source/data/norm provenance and code/config snapshots in run directory. GPU smoke snapshot predates optimizer-retention support; it has one final checkpoint only.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_torch --steps 2 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-000/full-gpu-sft-v3-001
```

### EXP-000 / official GPU wrapper tests and review / 2026-09-09
`audit/tests_a100_final.log`:20 passed,1 skipped,7.55s. New regression test verifiesN completed updates saves checkpointN, notN-1; retention test preserves all policy weights and latest optimizer. Static review found no concrete blockers in source/normalizer/pretrained guards or official GPU save callback. Real GPU two-step run separately confirmed finite training/checkpoint output.

### EXP-000/aligned-zero-gpu-v3-001 / 2026-09-09T21:22:24.367570+00:00
FAILED before model/environment creation: PyTorch rejected integer memory fraction1 when the requested budget reached total device capacity. Replaced with a validated float-valued helper; full-device and over-cap boundaries now tested. No episodes, no SR/ΔSR. Wall0.15152046829462051s; allocated0 bytes, reserved0 bytes; sampled device peak4MiB. A100, torch2.7.1+cu128/CUDA12.8; checkpoint not loaded. Revision/dirty tree/config and traceback in run directory.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-000/full-gpu-sft-v3-001/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-000/aligned-zero-gpu-v3-001
```
