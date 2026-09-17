# pi0 D0/D1 sanity audit: evaluation videos

These clips show the first two successes and first two failures in seed order for
each model and task, where available. They illustrate observed behavior; this
outcome-based sample does not estimate success rates. The full 50-episode results
per task remain in the [main report](PI0_D0_D1_SANITY_REPORT.md).
Notes describe sampled views spanning each clip, including its final observation;
they do not infer the policy's intent or hidden causes.

Each video shows **agentview on the left and wrist view on the right**, using the
same two cameras as the policy, downsampled to 128 pixels per view for recording.
Playback is 20 frames/second. A clip contains the pre-action observations and the
final post-action observation: exactly **episode length + 1 frames**.

The video requirement arrived after primary step 0 and the official control had
finished. Their selected episodes were replayed with identical seed, outcome,
length, reset hash, trajectory hash and image hash; original result rows were
preserved. The other clips were captured during their original evaluations.

All 38 linked MP4s were checked after copying: each exists, is nonempty,
decodes fully, has the expected frame count, matches its recorded seed/outcome,
and matches its source checksum. Total video size: 5.97 MiB.
[Machine-readable verification](pi0_audit/video_verification.json).

| Model/checkpoint | Task | Seed | Outcome | Video path | Short note |
|---|---|---:|---|---|---|
| Pre-SFT (official weights + D0 statistics) | D0 | 9000 | Failure | [Video](pi0_audit/videos/step0000_D0_seed9000_failure.mp4) | Arm moves leftward and toward the front of the table; the center bowl remains on the table and the plate stays empty. |
| Pre-SFT (official weights + D0 statistics) | D0 | 9001 | Failure | [Video](pi0_audit/videos/step0000_D0_seed9001_failure.mp4) | Arm sweeps left across the scene without transferring the center bowl to the plate. |
| Pre-SFT (official weights + D0 statistics) | D1 | 9000 | Failure | [Video](pi0_audit/videos/step0000_D1_seed9000_failure.mp4) | Arm moves across the bowl area and toward the front of the table; the bowl beside the plate remains on the table. |
| Pre-SFT (official weights + D0 statistics) | D1 | 9001 | Failure | [Video](pi0_audit/videos/step0000_D1_seed9001_failure.mp4) | Arm sweeps across the bowl arrangement; the target bowl stays beside the plate and the plate remains empty. |
| 500 D0 updates | D0 | 9000 | Failure | [Video](pi0_audit/videos/step0500_D0_seed9000_failure.mp4) | Gripper approaches and remains over the center bowl; the bowl stays on the table and the plate remains empty. |
| 500 D0 updates | D0 | 9001 | Failure | [Video](pi0_audit/videos/step0500_D0_seed9001_failure.mp4) | Gripper lowers beside the center bowl; no transfer to the plate is visible. |
| 500 D0 updates | D0 | 9002 | Success | [Video](pi0_audit/videos/step0500_D0_seed9002_success.mp4) | Gripper approaches the center bowl, carries it toward the plate, and lowers it onto the plate. |
| 500 D0 updates | D0 | 9018 | Success | [Video](pi0_audit/videos/step0500_D0_seed9018_success.mp4) | Gripper lifts the center bowl, moves it toward the plate, and lowers it onto the plate. |
| 500 D0 updates | D1 | 9000 | Failure | [Video](pi0_audit/videos/step0500_D1_seed9000_failure.mp4) | Gripper lowers into the empty center area; the bowl beside the plate stays on the table and no placement occurs. |
| 500 D0 updates | D1 | 9001 | Failure | [Video](pi0_audit/videos/step0500_D1_seed9001_failure.mp4) | Gripper moves down over empty space near the center; the bowl beside the plate remains off the plate. |
| 1,000 D0 updates | D0 | 9000 | Failure | [Video](pi0_audit/videos/step1000_D0_seed9000_failure.mp4) | Gripper approaches the center bowl; the bowl remains on the table and the plate stays empty. |
| 1,000 D0 updates | D0 | 9001 | Failure | [Video](pi0_audit/videos/step1000_D0_seed9001_failure.mp4) | Gripper moves down around the center bowl and rises again; no transfer onto the plate is visible. |
| 1,000 D0 updates | D0 | 9006 | Success | [Video](pi0_audit/videos/step1000_D0_seed9006_success.mp4) | Gripper lifts the center bowl, moves it across the table, and lowers it onto the plate. |
| 1,000 D0 updates | D0 | 9031 | Success | [Video](pi0_audit/videos/step1000_D0_seed9031_success.mp4) | Gripper picks up the center bowl, carries it toward the plate, and lowers it onto the plate. |
| 1,000 D0 updates | D1 | 9000 | Failure | [Video](pi0_audit/videos/step1000_D1_seed9000_failure.mp4) | Gripper lowers over the empty center area; the bowl beside the plate stays on the table. |
| 1,000 D0 updates | D1 | 9001 | Failure | [Video](pi0_audit/videos/step1000_D1_seed9001_failure.mp4) | Gripper moves over empty table space as the plate enters the wrist view; the bowl remains beside the plate. |
| 2,000 D0 updates | D0 | 9000 | Success | [Video](pi0_audit/videos/step2000_D0_seed9000_success.mp4) | Gripper lifts the center bowl and carries it onto the plate. |
| 2,000 D0 updates | D0 | 9001 | Failure | [Video](pi0_audit/videos/step2000_D0_seed9001_failure.mp4) | Gripper carries the center bowl near the plate, but the bowl remains offset from the plate center at the end. |
| 2,000 D0 updates | D0 | 9002 | Failure | [Video](pi0_audit/videos/step2000_D0_seed9002_failure.mp4) | Gripper moves from the center area toward the plate while the center bowl remains on the table. |
| 2,000 D0 updates | D0 | 9004 | Success | [Video](pi0_audit/videos/step2000_D0_seed9004_success.mp4) | Gripper approaches the center bowl, lifts it, and moves it onto the plate. |
| 2,000 D0 updates | D1 | 9000 | Failure | [Video](pi0_audit/videos/step2000_D1_seed9000_failure.mp4) | Gripper lowers over the empty center area and then moves near the plate; the bowl remains beside the plate. |
| 2,000 D0 updates | D1 | 9001 | Failure | [Video](pi0_audit/videos/step2000_D1_seed9001_failure.mp4) | Arm moves from the empty center toward the plate area; the bowl remains outside the plate. |
| 3,001 D0 updates | D0 | 9000 | Success | [Video](pi0_audit/videos/step3001_D0_seed9000_success.mp4) | Gripper lifts the center bowl, carries it toward the plate, and lowers it onto the plate. |
| 3,001 D0 updates | D0 | 9001 | Success | [Video](pi0_audit/videos/step3001_D0_seed9001_success.mp4) | Gripper reaches the center bowl and transfers it onto the plate. |
| 3,001 D0 updates | D0 | 9004 | Failure | [Video](pi0_audit/videos/step3001_D0_seed9004_failure.mp4) | Gripper carries the center bowl to the plate area but is still holding it near the plate at the end. |
| 3,001 D0 updates | D0 | 9008 | Failure | [Video](pi0_audit/videos/step3001_D0_seed9008_failure.mp4) | Gripper lifts the center bowl and carries it near the plate; the bowl remains in the gripper at the end. |
| 3,001 D0 updates | D1 | 9000 | Failure | [Video](pi0_audit/videos/step3001_D1_seed9000_failure.mp4) | Gripper lowers over the empty center area, then moves toward the plate without a bowl; the target bowl stays beside the plate. |
| 3,001 D0 updates | D1 | 9001 | Failure | [Video](pi0_audit/videos/step3001_D1_seed9001_failure.mp4) | Gripper moves from the empty center toward the plate; the wrist view shows the plate while the bowl remains on the table beside it. |
| LoRA initialization (0 updates) | D0 | 9000 | Failure | [Video](pi0_audit/videos/lora_initialization_D0_seed9000_failure.mp4) | Arm moves leftward across the scene; the center target bowl stays on the table and the plate remains empty. |
| LoRA initialization (0 updates) | D0 | 9001 | Failure | [Video](pi0_audit/videos/lora_initialization_D0_seed9001_failure.mp4) | Arm passes to the left of the bowl arrangement; the target bowl remains on the table and no placement is visible. |
| LoRA initialization (0 updates) | D1 | 9000 | Failure | [Video](pi0_audit/videos/lora_initialization_D1_seed9000_failure.mp4) | Arm lowers toward the front of the table; the bowl remains off the plate at the end. |
| LoRA initialization (0 updates) | D1 | 9001 | Failure | [Video](pi0_audit/videos/lora_initialization_D1_seed9001_failure.mp4) | Arm sweeps over the bowl area and then rises; no bowl is left on the plate. |
| Official pi0.5 LIBERO (positive control) | D0 | 9000 | Success | [Video](pi0_audit/videos/official_libero_D0_seed9000_success.mp4) | Gripper lifts the center bowl and transfers it onto the plate. |
| Official pi0.5 LIBERO (positive control) | D0 | 9001 | Success | [Video](pi0_audit/videos/official_libero_D0_seed9001_success.mp4) | Gripper lifts the center bowl and transfers it onto the plate. |
| Official pi0.5 LIBERO (positive control) | D1 | 9000 | Success | [Video](pi0_audit/videos/official_libero_D1_seed9000_success.mp4) | Gripper picks up the bowl beside the plate and lowers it onto the plate. |
| Official pi0.5 LIBERO (positive control) | D1 | 9001 | Success | [Video](pi0_audit/videos/official_libero_D1_seed9001_success.mp4) | Gripper picks up the bowl beside the plate and lowers it onto the plate. |
| Official pi0.5 LIBERO (positive control) | D1 | 9004 | Failure | [Video](pi0_audit/videos/official_libero_D1_seed9004_failure.mp4) | Gripper lifts the bowl beside the plate and carries it over the plate area; the bowl remains in the gripper at the end. |
| Official pi0.5 LIBERO (positive control) | D1 | 9006 | Failure | [Video](pi0_audit/videos/official_libero_D1_seed9006_failure.mp4) | Gripper lifts the bowl beside the plate and carries it over the plate area; the bowl remains in the gripper at the end. |

## Outcome coverage

An outcome with zero available episodes has no example to save. Categories with
one episode retain that episode; categories with two or more retain two.

| Model/checkpoint | Task | Outcome | Available among 50 episodes | Videos saved |
|---|---|---|---:|---:|
| Pre-SFT (official weights + D0 statistics) | D0 | Success | 0/50 | 0 |
| Pre-SFT (official weights + D0 statistics) | D0 | Failure | 50/50 | 2 |
| Pre-SFT (official weights + D0 statistics) | D1 | Success | 0/50 | 0 |
| Pre-SFT (official weights + D0 statistics) | D1 | Failure | 50/50 | 2 |
| 500 D0 updates | D0 | Success | 3/50 | 2 |
| 500 D0 updates | D0 | Failure | 47/50 | 2 |
| 500 D0 updates | D1 | Success | 0/50 | 0 |
| 500 D0 updates | D1 | Failure | 50/50 | 2 |
| 1,000 D0 updates | D0 | Success | 2/50 | 2 |
| 1,000 D0 updates | D0 | Failure | 48/50 | 2 |
| 1,000 D0 updates | D1 | Success | 0/50 | 0 |
| 1,000 D0 updates | D1 | Failure | 50/50 | 2 |
| 2,000 D0 updates | D0 | Success | 25/50 | 2 |
| 2,000 D0 updates | D0 | Failure | 25/50 | 2 |
| 2,000 D0 updates | D1 | Success | 0/50 | 0 |
| 2,000 D0 updates | D1 | Failure | 50/50 | 2 |
| 3,001 D0 updates | D0 | Success | 38/50 | 2 |
| 3,001 D0 updates | D0 | Failure | 12/50 | 2 |
| 3,001 D0 updates | D1 | Success | 0/50 | 0 |
| 3,001 D0 updates | D1 | Failure | 50/50 | 2 |
| LoRA initialization (0 updates) | D0 | Success | 0/50 | 0 |
| LoRA initialization (0 updates) | D0 | Failure | 50/50 | 2 |
| LoRA initialization (0 updates) | D1 | Success | 0/50 | 0 |
| LoRA initialization (0 updates) | D1 | Failure | 50/50 | 2 |
| Official pi0.5 LIBERO (positive control) | D0 | Success | 50/50 | 2 |
| Official pi0.5 LIBERO (positive control) | D0 | Failure | 0/50 | 0 |
| Official pi0.5 LIBERO (positive control) | D1 | Success | 47/50 | 2 |
| Official pi0.5 LIBERO (positive control) | D1 | Failure | 3/50 | 2 |
