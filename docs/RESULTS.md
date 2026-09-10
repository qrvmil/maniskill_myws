# RESULTS — PLD LIBERO

## Current headline result
Source-only full-model pi0 alignment completed3000 updates on the actual A10080GB. Its frozen checkpoint scored **3/10 source-validation successes (30%)**, versus0/10 for the100-update pilot on the same validation seeds. This is exploratory single-seed base-policy validation, not a residual-gain or unseen-task result. All24 integration/contract tests passed. Selected-checkpoint zero-residual verification passed exactly on two pairs; successful-base collection completed48/100 with6484 validated transitions. Real-buffer A smoke completed2 Cal-QL and8 SAC updates; its paired D0 execution check scored base1/2 versus residual0/2 (−50pp). OTF smoke also passed; main50000-step source training is running. **No main-specialist cross-task transfer result exists.**

## Base policy
Corrected corpus:50 source demonstrations,5832 temporally aligned observation/action pairs. All earlier same-index checkpoints are excluded. The3000-update checkpoint is frozen for the next offline/RL stages based only on source validation seeds2000–2009. Successes occurred at2000,2004,2006; seven time-limit failures. Mean length192.2 steps. Successful frozen-base collection completed48/100 attempts,6484 transitions, all on training seeds1000–1099. No demonstrations were used in residual replay. User authorized A100 usage; current experiments no longer impose16GiB.

## Source residual training
EXP-002: real-buffer A and OTF smoke gates passed. Main B completed all1000 Cal-QL warm-start updates with finite losses and entered the registered50000-step online phase. All100 base-action warmup episodes completed17924 steps and48 successes, with exact reset/trajectory/RGB matches to the frozen-base collection. Active OTF residual interaction now uses the remaining32076 steps of the fixed50000-step budget. Main final checkpoint and performance remain PENDING. No distillation will be run.

## Cross-task transfer
| Source | Target | Distance | SR base | SR + residual | ΔSR | Episodes | Seed(s) |
|---|---|---|---|---|---|---|---|
| bowl center → plate | source | D0 | PENDING | PENDING | PENDING | 0 | PENDING |
| bowl center → plate | frozen design ladder | D1–D5 | PENDING | PENDING | PENDING | 0 | PENDING |

## Gain vs distance
PENDING. No aggregate from unrun or dummy-policy episodes.

## Negative transfer cases
Primary cross-task cases: PENDING. The eight-update A engineering fixture worsened source D0 success from1/2 to0/2 (−50pp) on seeds3000/3001. This observed negative same-task correction is retained; two episodes and eight training updates do not establish the behavior of the planned main specialist.

## Runtime / memory
All measurements below are from the actual A100 80 GB. The user authorized its full capacity; they are not RTX 5080 results.

| Completed run | Peak PyTorch allocation | Sampled device peak | Measured time |
|---|---:|---:|---|
| Corrected CPU-offloaded full SFT, 2 updates | 12.69 GiB | 13,739 MiB | 12.11 s second update; 148.16 s total |
| Official GPU full SFT, 100 updates | 29.99 GiB | 32,063 MiB | 0.9 s median logged update interval; 291.18 s total |
| Source validation of GPU-100 checkpoint, 10 episodes | 6.67 GiB | 8,179 MiB | 23.09 s mean episode; 9.53 environment steps/s excluding setup; 313.02 s total |
| Official GPU full SFT, 3000 updates | 29.99 GiB | 32,065 MiB | 2825.93 s total |
| Source validation of GPU-3000 checkpoint, 10 episodes | 6.67 GiB | 8,179 MiB | 20.88 s mean episode; 290.55 s total |
| Real-buffer A smoke, 2 Cal-QL + 8 SAC | 6.76 GiB combined; Cal-QL 6.75 GiB; SAC 6.66 GiB | 8,315 MiB | 93.48 s total; 0.10 s second Cal-QL update |
| Main Cal-QL completed phase, 1000 updates; online run continues | 8.90 GiB including frozen VLA | Whole-run total pending | 98.91 s summed update time; 0.098 s median update |
| Combined full-budget frozen VLA + residual training | PENDING | PENDING | PENDING |

The source-validation run recorded 60.89 s for provenance checks and 14.65 s for model loading. Per-phase peaks include persistent model allocations. Raw metadata and memory samples are preserved per run; low-impact dummy-update measurements remain engineering diagnostics in the ledger.

## Failed runs

| Run | Failure / limitation | Resolution or status |
|---|---|---|
| EXP-000/full_sft_probe/run-001 | Synthetic image layout BHWC instead of BCHW | Corrected probe layout |
| EXP-000/full_sft_probe/run-002 | Full GPU AdamW OOM under 16 GiB cap | CPU offload passed; user later authorized full A100 |
| EXP-000/lora-probe-001 | Official JAX LoRA OOM in constrained pool | Not selected; full GPU SFT now works on A100 |
| EXP-000/smoke-001 | CLI filename shadowed LIBERO import | Renamed entrypoint |
| EXP-000/smoke-002/003/005 | RGB nondeterminism despite identical actions/physics | Disabled offscreen MSAA; subsequent full zero checks passed |
| EXP-000/full-cpu-sft-001/002 | Meta initialization broke tied weights/nonpersistent positional buffers | Safe CPU initialization and strict loading |
| EXP-000/aligned-zero-001 | Missing easydict dependency | Installed and added preflight/setup support |
| EXP-001/sft-pilot-100 | Interrupted at 46 updates after native observation/action offset was identified | Corrected corpus; earlier checkpoints excluded from scientific use |
| EXP-000/aligned-zero-gpu-v3-001 | Full-capacity memory fraction returned integer instead of float | Fixed and regression-tested; rerun passed |
| EXP-001/source-validation-gpu-100 | 0/10 successes after short SFT pilot | 3000-update alignment completed; source validation next; no offline-buffer substitution |

Original errors, OOM diagnostics and exact commands remain in the append-only ledger below. Initial shell/checkout setup failures and conversion verification's unused-random-head mismatch are also recorded there.

## Open questions / next experiments
Highest priority: complete paired smoke/OTF gates, then train the registered50000-step source specialist and evaluate its fixed checkpoint across D0–D5. Do not use leaked full-LIBERO checkpoints to bypass this gate.

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

### EXP-000/aligned-zero-gpu-v3-002 / 2026-09-09T21:24:54.270656+00:00
Both zero-residual pairs passed with exact RGB/action/full92D-physics equality. SourceD0, seeds2000/2001, four220-step environment episodes. Base0/2, zero residual0/2; SR_base0.0, SR_zero0.0, ΔSR_zero0.0. This is not a learned correction or transfer result. GPU-trained2-update checkpoint `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-000/full-gpu-sft-v3-001/checkpoints/pi0_libero_seen_full_torch/EXP-001/2`.

Wall236.0753170941025s; allocated7162114048 bytes, reserved7275020288 bytes; sampled device peak8179MiB; CPU RAM15378808832 bytes. A100, torch2.7.1+cu128/CUDA12.8; revision `a6274bdabf024ae270689000c39d587d066de311`, dirty status/config/code snapshots in run. Earlier CLI timing excluded one initial checksum pass; this run still used redundant checksum passes inside evaluation. Subsequent commands verify provenance once inside timed artifacts before model loading, preserving guards and measuring separate validation/load durations.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-000/full-gpu-sft-v3-001/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-000/aligned-zero-gpu-v3-002
```

### EXP-001/sft-gpu-v3-100 / STARTED
Supervisor `pld_libero_sft` started the preregistered100-update source-only official GPU pilot after its2-update checkpoint passed zero equivalence. Fresh same official pretrained weights, corrected5832-pair source corpus, batch1, training seed0, GPU AdamW, A10080GB. Checkpoint/success/runtime final values PENDING. Exact wrapper and supervisor config saved under EXP-000/audit/supervisor_sft_gpu_100.sh and supervisor_sft.conf. Run metadata/code snapshots record the actual execution.

### EXP-001/sft-gpu-v3-100 / COMPLETED / 2026-09-09T21:29:31.784494+00:00
All100 source-only official GPU SFT updates completed with finite logged losses/gradients. Loss range0.0101–0.8328; final logged loss0.2462. Native loss/gradient logs are rounded. Median logged per-update interval0.9s; total wall291.18468353897333s includes model initialization/save/provenance hashing. All3501372176 parameters eligible. No environment episodes during training; scientific success/ΔSR PENDING.

Peak allocated32200932352 bytes (29.989GiB), reserved32623296512 bytes; sampled device peak32063MiB; CPU RAM16304824320 bytes. A10080GB, torch2.7.1+cu128, CUDA12.8, sourceD0/training seed0. Checkpoint `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-001/sft-gpu-v3-100/checkpoints/pi0_libero_seen_full_torch/EXP-001/100`. Revision `aa4c6c49c0448b88bee99cafcfe68a0f0b18f0cd`, dirty status/code/config/protocol snapshots and per-step logs in run directory. The source-validation10-episode run is launched separately; no unseen task executed.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_torch --cpu-threads 16 --steps 100 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-001/sft-gpu-v3-100
```

### EXP-001/source-validation-gpu-100 / 2026-09-09T21:35:15.209769+00:00
Completed10 source-only validation episodes, seeds2000–2009, taskID2/D0, all220 steps. Successes0/10; SR_base0.0. No learned residual evaluated; SR_residual/ΔSR PENDING. All failures reached the time limit; no stronger physical diagnosis inferred. Checkpoint `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-001/sft-gpu-v3-100/checkpoints/pi0_libero_seen_full_torch/EXP-001/100`, training seed0. This is exploratory source validation, not final held-out D0 or cross-task evaluation. No offline collection was attempted because this pilot produced no validation successes.

Wall313.0173181798309s including60.88972958177328s provenance verification and14.650440579280257s model loading. Mean episode wall23.094131148047744s. Base-inference peak allocated7162114048 bytes; overall allocated7162114048 bytes, reserved7275020288 bytes; sampled device peak8179MiB; CPU RAM15376179200 bytes. A100, torch2.7.1+cu128/CUDA12.8; revision `a6bf473d9f8639008811fff99036e4c5267c8a24` and dirty/config/code snapshots in run.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py base --validation --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-100/alignment_manifest.json --episodes 10 --output outputs/pld_libero/EXP-001/source-validation-gpu-100
```

### EXP-001/sft-gpu-v3-3000 / REGISTERED AFTER SOURCE-ONLY PILOT
The100-step pilot scored0/10, so proceed to the previously planned3000-update alignment budget. Fresh identical pretrained pi0 weights; same50 source demonstrations/5832 corrected pairs, batch1, seed0, official GPU AdamW, sameLR schedule and source-only statistics. No unseen data or task success rates influenced this decision. Save every500/final and retain latest optimizer plus all policy weights. Projected45–55minutes from observed0.9s/update plus initialization/save/checksum overhead; projection is not a measured result. Subsequent source-only validation again uses2000–2009. Frozen split unchanged.

### EXP-001/sft-gpu-v3-100-figures-002 / 2026-09-09
Exported reviewed PNG/SVG training-loss plot from all100 completed pilot updates, raw update points plus trailing20-update median. Native logged values and adjacent learning-rate/gradient/runtime fields retained in sft_input.csv; source metadata/manifest hashes in chart_provenance.json. The first draft's legend overlapped an outlier; it was superseded by figures-002 with the legend outside the plot area. No transfer chart is generated without actual completed learned-residual evaluations.

Command: `PYTHONPATH=src /venv/main/bin/python scripts/pld/plot_libero.py sft --runs outputs/pld_libero/EXP-001/sft-gpu-v3-100 --output outputs/pld_libero/EXP-001/sft-gpu-v3-100-figures-002`.

[Reviewed source-SFT loss figure](../outputs/pld_libero/EXP-001/sft-gpu-v3-100-figures-002/sft_loss.png). This figure does not establish task success or residual gain.

### EXP-001/sft-gpu-v3-100-figures-003 / 2026-09-09
Re-exported the reviewed figure through the pinned OpenPI environment (Matplotlib 3.10.3, NumPy 1.26.4). The figure was visually inspected; all 100 points, including the largest loss at update 74, remain visible. Exact plotting command, renderer versions, plotter hash/snapshot and source hashes are in chart_provenance.json. This is the canonical pilot figure: [SFT loss](../outputs/pld_libero/EXP-001/sft-gpu-v3-100-figures-003/sft_loss.png). No numerical input changed from figures-002.

### EXP-000 / GPU restart reproducibility / 2026-09-09
`audit/gpu_restart_first100.json`: the first 100 completed update losses, gradient norms and learning rates in the fresh 3000-step run exactly match the completed 100-step pilot at native logged precision. This is a rounded-log comparison, not a claim of bitwise checkpoint equality.

### EXP-000 / pre-RL entropy-coordinate audit / 2026-09-09T22:21:05.385292+00:00
Main B now interprets the paper target -7/2 in unit residual coordinates. Because the inherited actor includes the scale Jacobian, the actual scaled-density target is -8.352030263919616 at scale0.5. The paper does not establish this coordinate convention; author-code equivalence remains unverified. No real residual training, successful-base collection or unseen evaluation has run, so this amendment precedes scientific RL results. SAC implementation remains unchanged. A temperature-gradient invariance test first failed (difference4.851989) with the old unshifted value and then passed with the corrected configuration: `audit/test_entropy_red.log` and `audit/test_entropy_green.log`. Numerical tolerance0.002 accommodates the inherited Jacobian stabilizer near saturation.

### EXP-001/sft-gpu-v3-3000 / COMPLETED / 2026-09-09T21:41:28.475999+00:00
Completed all3000 full-model updates on the corrected source-only corpus; all3501372176 parameters trainable, batch1, seed0, official GPU AdamW. Wall2825.9334021657705s; peak allocated32200932352 bytes, reserved32623296512 bytes; sampled device peak32065MiB; host RSS16233906176 bytes. A10080GB, torch2.7.1+cu128/CUDA12.8. Revision `a6bf473d9f8639008811fff99036e4c5267c8a24`; dirty status `M docs/RESULTS.md`, implementation snapshots retained. All3000 logged losses/gradient norms finite. Final loss0.0751, gradient norm1.19; median logged update interval0.7s. No policy success inferred from training loss.

Checkpoint: `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-001/sft-gpu-v3-3000/checkpoints/pi0_libero_seen_full_torch/EXP-001/3000`. All policy checkpoints retained; only latest optimizer state retained. Source-validation SR, residual SR and gain PENDING at completion.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/align_libero.py train --source-h5 /workspace/datasets/libero_seen/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 --repo-id local/pld_libero_bowl_v3 --dataset-root /workspace/datasets/lerobot/local/pld_libero_bowl_v3 --method full_torch --cpu-threads 16 --steps 3000 --pytorch-base-checkpoint /workspace/checkpoints/pi0_base_pytorch_v2 --output outputs/pld_libero/EXP-001/sft-gpu-v3-3000
```

### EXP-000 / full integration verification / 2026-09-09T22:29:47.559275+00:00
`PLD_LIBERO_INTEGRATION=1 scripts/pld/libero_python.sh -m pytest -q tests/test_pld_libero.py` passed24 tests with3 upstream robosuite deprecation warnings in25.54s. Includes actual LIBERO reset/step/zero-trajectory equivalence, visual replay/Cal-QL/SAC synthetic updates, checkpoint determinism, leakage guards and entropy-coordinate invariance. Log `audit/tests_final_integration_24.log`. This does not substitute for real successful-base-buffer RL validation.

### EXP-001/sft-gpu-v3-3000-figures-001 / 2026-09-09T22:29:47.559287+00:00
Exported and visually inspected all3000 completed update losses with a trailing20-update median. [SFT loss figure](../outputs/pld_libero/EXP-001/sft-gpu-v3-3000-figures-001/sft_loss.png); SVG, raw CSV and provenance are adjacent. Command: `scripts/pld/libero_python.sh scripts/pld/plot_libero.py sft --runs outputs/pld_libero/EXP-001/sft-gpu-v3-3000 --output outputs/pld_libero/EXP-001/sft-gpu-v3-3000-figures-001`. No task-success inference from this curve.

### EXP-001/source-validation-gpu-3000 / COMPLETED / 2026-09-09T22:29:31.815381+00:00
Source `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, taskID2, D0 source-validation seeds2000–2009, training seed0. Base successes3/10, SR_base0.3, mean length192.2; successful seeds2000 (151steps),2004 (109),2006 (122). Seven failures reached the220-step cap. Residual not evaluated: SR_residual/ΔSR PENDING. Freeze this checkpoint for subsequent residual stages; only source validation informed this selection. No unseen task was evaluated.

Wall290.54549937322736s; provenance61.05509691871703s; base load14.467890746891499s; mean episode wall20.875783143751324s. Peak base inference/overall allocation7162114048 bytes, reserved7275020288 bytes, sampled device8179MiB, host RSS15379156992 bytes. A10080GB, torch2.7.1+cu128/CUDA12.8. Revision `4253765dbd5ce4e35413bc021c4ffae126050d3a`, dirty status `M docs/RESULTS.md`; complete config/code snapshots in run.

Checkpoint: `/workspace/State-Estimation/maniskill_myws/outputs/pld_libero/EXP-001/sft-gpu-v3-3000/checkpoints/pi0_libero_seen_full_torch/EXP-001/3000`.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py base --validation --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --episodes 10 --output outputs/pld_libero/EXP-001/source-validation-gpu-3000
```

### EXP-001/zero-gpu-3000 / COMPLETED / 2026-09-09T22:34:37.729149+00:00
Selected frozen GPU3000 base, source taskID2/D0, validation seeds2000/2001, training seed0. Two paired conditions/four actual rollouts: base1/2, zero-residual1/2, both50%, ΔSR0 for the **zero correction only**. All action/RGB/next-RGB/full-physics maximum errors exactly0. Successful episodes151steps, timeouts220. This verifies the integration gate; it is not learned-residual transfer. Historical metadata `episodes=2` denotes paired conditions; newer evaluation metadata additionally records total `rollout_episodes`.

Wall161.2530006673187s; allocation7162114048 bytes, reserved7275020288 bytes; sampled device8179MiB, host RSS15381135360 bytes. A10080GB, torch2.7.1+cu128/CUDA12.8. Revision `4253765dbd5ce4e35413bc021c4ffae126050d3a`, dirty `M docs/RESULTS.md`; source checkpoint/config and exact zero checks in run.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py zero --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --episodes 2 --output outputs/pld_libero/EXP-001/zero-gpu-3000
```

### EXP-000 / residual checkpoint reporting guard / 2026-09-09
Pre-training review found that a short real RL-smoke/A checkpoint could previously be evaluated under the main B config because only source/seed/action/image contracts were checked. Added explicit training-regimen and completed-budget checks plus actual SAC configuration in checkpoint/evaluation/summary metadata. This prevents reporting8-step smoke weights as50000-step main weights; intermediate checkpoints remain saved but primary evaluation requires the registered final budget. Regression tests failed before the guard and passed afterward: `audit/test_regimen_red.log`, `audit/test_regimen_green.log`, `audit/test_summary_regimen_red.log`, `audit/test_regimen_combined_green.log`. No actual residual checkpoint or unseen result existed before the fix.

### EXP-000 / checkpoint-regimen regression suite / 2026-09-09
` scripts/pld/libero_python.sh -m pytest -q tests/test_pld_libero.py` completed24 passed/1 integration test skipped in20.56s (`audit/tests_regimen_full.log`). The actual simulator24-test suite had already passed before these reporting-guard-only changes; no concurrent GPU integration test was run during scientific collection.

### EXP-001/offline-gpu-3000 / RUNNING / 2026-09-09T22:37:41.242974+00:00
Collect successful rollouts of the frozen selected GPU3000 source-only pi0 on source taskID2/D0, training environment seeds1000–1099, model/noise training configuration seed0. Target50 successes, cap100 attempts; no demonstrations or foreign replay permitted. Final attempts/successes/transitions/base SR/runtime/memory PENDING until completion. Partial attempt records are persisted after every episode in `eval/collection.json`. Managed by supervisor `pld_libero_collect`. Revision `472fffabad990df44d84feaf07186496c7936baf`, dirty status `clean`; A10080GB, torch2.7.1+cu128/CUDA12.8.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py collect --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-gpu-3000/eval/zero_equivalence.json --successes 50 --max-attempts 100 --output outputs/pld_libero/EXP-001/offline-gpu-3000
```

### EXP-001/offline-gpu-3000 / COMPLETED
Actual source-base collection:100 attempts on training seeds1000–1099,48 successful trajectories,6484 transitions, SR_base0.48. Target50 successes was not reached within the100-attempt cap; retain the actual48, without duplicates or additional seeds. Replay contains only these actual frozen-base successes, two uint8 RGB views and proprioception/base/executed/next-state actions; sparse terminal success reward and discounted MC return with gamma0.99. Source remains LIBERO spatial taskID2/D0; selected GPU3000 base/seed0. No residual was applied or evaluated; ΔSR PENDING.

Wall2455.661007400602s; mean episode21.733031460680067s. Process peak allocated7162114048 bytes/reserved7275020288 bytes; host RSS15379738624 bytes. Sampled total device peak8601MiB is **not isolated**: the fail-closed CLI unit test launched CUDA-enabled commands without an alignment manifest, briefly adding a context during collection (peak at235.04–245.55s). The process-specific PyTorch peak remains valid; isolated prior base inference measured8179MiB. The CLI test now selects CPU and hides CUDA. A10080GB, torch2.7.1+cu128/CUDA12.8; collection revision `472fffabad990df44d84feaf07186496c7936baf`, clean. Full command is in the preceding registered entry and metadata.

### EXP-000/real-buffer-validation-gpu3000 / COMPLETED / 2026-09-09T23:19:50.517803+00:00
Independent replay audit passed100 attempts/48 trajectories/6484 transitions. Recomputed discounted returns differ by at most2.9755748531812287e-08; all nonterminal next states, next base actions and next images exactly match the subsequent stored transition. Actions equal frozen-base actions and remain bounded; terminal masks/rewards and provenance match collection. Buffer SHA256 `b5bf1a5286cff1d5a0e18390c59e46349a0e0d88343363071c32819737bc43b1`; uint8 RGB arrays occupy1274806272 bytes in CPU memory. Wall12.702235711738467s. Script snapshot, input hashes, seeds, source and git state in verification.json/verifier.py.

```sh
scripts/pld/libero_python.sh outputs/pld_libero/EXP-000/audit/verify_collected_buffer.py --collection outputs/pld_libero/EXP-001/offline-gpu-3000 --output outputs/pld_libero/EXP-000/real-buffer-validation-gpu3000
```

### EXP-002/rl-smoke-gpu3000-seed0 / COMPLETED / 2026-09-09T23:20:26.022649+00:00
Real48-trajectory base buffer, source taskID2/D0, residual training seed0, config `anchor_bowl_rl_smoke.json`: batch2,2 Cal-QL updates (2 candidates),8 online SAC updates, warmup0, source seed1000. All10 updates finite; first Cal-QL q_loss56.11315155029297, second58.29925537109375; final SAC q_loss12.125715255737305, actor_loss4.4264140129089355, alpha0.9991021752357483. One8-step engineering rollout; this is not a source-performance experiment. Primary SR/gain PENDING.

Wall93.4805209748447s; combined peak allocated7258752512 bytes, reserved7405043712 bytes, sampled device8315MiB. Per-phase peaks (include persistent frozen pi0/residual allocations): Cal-QL7252179968 bytes, SAC7151380992 bytes, base inference7258752512 bytes. Second Cal-QL update0.10150586068630219s; SAC update median0.04774454887956381s. Exact individual memory/time/loss fields in logs/updates.jsonl. Host RSS15377190912 bytes. A10080GB, torch2.7.1+cu128/CUDA12.8; revision `a8475ff504ceda9c0fe06872aa5abf251dc9d064`, clean.

Checkpoint: `outputs/pld_libero/EXP-002/rl-smoke-gpu3000-seed0/checkpoints/residual_step_8.pt`. Actual training regimen and SAC configuration are in its sidecar; it cannot pass as a main50000-step checkpoint.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_rl_smoke.json --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-gpu-3000/eval/zero_equivalence.json --offline-buffer outputs/pld_libero/EXP-001/offline-gpu-3000/offline.npz --output outputs/pld_libero/EXP-002/rl-smoke-gpu3000-seed0
```

### EXP-000 / CPU-only negative CLI test / 2026-09-09
Corrected the test harness to use a CPU configuration and `CUDA_VISIBLE_DEVICES` empty for missing-alignment CLI probes. The production guard is unchanged. `CUDA_VISIBLE_DEVICES='' scripts/pld/libero_python.sh -m pytest -q tests/test_pld_libero.py -k scientific_commands` passed1 test/24 deselected in12.85s; `audit/test_cpu_only_cli_guard.log`. Earlier total-device peaks that overlapped this test include its extra CUDA context and should be treated as conservative whole-device measurements; process-specific PyTorch peaks remain separate.

### EXP-000/learned-residual-smoke-gpu3000-seed0 / COMPLETED / 2026-09-09T23:22:04.005267+00:00
Paired deterministic evaluation of the eight-update A fixture on source taskID2/D0, same seeds3000/3001 and initial states. Base1/2 (50%), residual0/2 (0%), ΔSR−0.5 (−50pp). Base lengths220/105, residual220/220; mean162.5 versus220. Base solved seed3001 and the residual failed it. Both residual episodes timed out. Four actual rollouts/two paired conditions. This is an engineering source execution check with no success threshold, explicitly excluded from the main50000-step transfer curve; the negative observation is not hidden. No unseen task was run.

Wall186.20190326310694s; allocated7193941504 bytes, reserved7300186112 bytes, sampled device8203MiB; host RSS15380393984 bytes. A10080GB, torch2.7.1+cu128/CUDA12.8; revision `a8475ff504ceda9c0fe06872aa5abf251dc9d064`, clean. Checkpoint `outputs/pld_libero/EXP-002/rl-smoke-gpu3000-seed0/checkpoints/residual_step_8.pt`, training seed0. Actual SAC/regimen/budget are recorded in summary and metadata.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py eval --config configs/pld_libero/anchor_bowl_rl_smoke.json --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-gpu-3000/eval/zero_equivalence.json --checkpoint outputs/pld_libero/EXP-002/rl-smoke-gpu3000-seed0/checkpoints/residual_step_8.pt --distance D0 --episodes 2 --output outputs/pld_libero/EXP-000/learned-residual-smoke-gpu3000-seed0
```

### EXP-002/rl-otf-smoke-gpu3000-seed0 / COMPLETED / 2026-09-09T23:25:14.689133+00:00
Real48-trajectory/6484-transition base buffer; source taskID2/D0, seed0, config `anchor_bowl_otf_smoke.json`. Completed2 Cal-QL and8 SAC updates with finite losses; OTF1 sampled residual plus the base candidate, hard-Q target backup, shifted main-B entropy target. Final SAC q_loss0.039978526532649994, actor_loss2.6800150871276855, alpha0.9991045594215393; reported backup candidate count2.0. One8-step source engineering rollout, no source-performance claim or unseen evaluation.

Wall93.18684179522097s; combined allocation7258752512 bytes, reserved7405043712 bytes, sampled device8315MiB. Per-phase memory and update wall times are in metadata/updates.jsonl. A10080GB, torch2.7.1+cu128/CUDA12.8; revision `a8475ff504ceda9c0fe06872aa5abf251dc9d064`, dirty `M docs/RESULTS.md
 M src/maniskill_myws/pld/libero_experiment.py
 M tests/test_pld_libero.py`. Checkpoint `outputs/pld_libero/EXP-002/rl-otf-smoke-gpu3000-seed0/checkpoints/residual_step_8.pt`. This fresh smoke does not initialize the main specialist.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_otf_smoke.json --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-gpu-3000/eval/zero_equivalence.json --offline-buffer outputs/pld_libero/EXP-001/offline-gpu-3000/offline.npz --output outputs/pld_libero/EXP-002/rl-otf-smoke-gpu3000-seed0
```

### EXP-002/source-otf-gpu3000-seed0 / RUNNING / 2026-09-09T23:26:52.417741+00:00
Start main B only after zero equivalence, actual replay verification, finite A smoke/paired execution and finite OTF smoke. Fixed source taskID2/D0, residual seed0;1000 Cal-QL updates (10 candidates), then50000 online steps, batch8,100 base-only warmup episodes with SAC updates inside that budget, OTF1 plus base candidate. Frozen selected GPU3000 base; actual offline48 successful trajectories/6484 transitions. Fresh residual initialization, no smoke weights reused. No unseen task has been evaluated. Main source performance, final checkpoint/runtime/VRAM and transfer metrics PENDING. Managed by supervisor `pld_libero_rl`. A10080GB, torch2.7.1+cu128/CUDA12.8; revision `e985c0720630fa08bd284748ecc86e985c453062`, dirty status `M docs/RESULTS.md`.

```sh
/workspace/State-Estimation/maniskill_myws/third_party/openpi/.venv/bin/python scripts/pld/run_libero.py train --config configs/pld_libero/anchor_bowl_otf.json --alignment-manifest outputs/pld_libero/EXP-001/sft-gpu-v3-3000/alignment_manifest.json --zero-report outputs/pld_libero/EXP-001/zero-gpu-3000/eval/zero_equivalence.json --offline-buffer outputs/pld_libero/EXP-001/offline-gpu-3000/offline.npz --output outputs/pld_libero/EXP-002/source-otf-gpu3000-seed0
```

### EXP-002/source-otf-gpu3000-seed0 / CAL-QL PHASE COMPLETED / observed 2026-09-09T23:31:28.325691+00:00
All1000 registered warm-start updates completed with finite losses. First q_loss72.4727554321289, last61.29702377319336; the conservative objective includes its density/log-sum-exp terms, so these values are not task-success scores. Summed update time98.91148052178323s, median0.09751475602388382s. Phase peak allocated9558611456 bytes (8.90GiB, includes resident frozen pi0), reserved9797894144 bytes. `calql_phase_summary.json` refers to the first1000 rows of logs/updates.jsonl. Online SAC is now running; this phase completion is not completion of the specialist experiment.

### EXP-002/source-otf-gpu3000-seed0 / WARMUP COMPLETED / observed 2026-09-10T00:23:33.530592+00:00
Completed100 registered base-only warmup episodes,17924 environment/SAC update steps,48 successes. All100 reset hashes, trajectory hashes (physics/actions/proprioception) and RGB-sequence hashes exactly match the corresponding source-base collection on seeds1000–1099, while residual SAC parameters were updating. `warmup_equivalence.json` binds the comparison to the actual collection file hash. This is a frozen-base/control reproducibility diagnostic, not residual improvement. The warmup consumes35.848% of the50000-step online budget;32076 steps remain for active OTF residual interaction. Main final metrics remain PENDING.
