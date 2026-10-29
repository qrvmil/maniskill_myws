# Source audit and reproduction boundaries

Read before any experiment implementation. This is an audit, not an approved new training configuration.

## Immutable sources

- Three adaptations: `qrvmil/maniskill_myws`, branch `exp/pi0-d0-d1-three-adaptations-20261020`, HEAD `48139c7ebba24d180ee2302091232f1b87cd92ec`. Read README, experiments/pi0_d0_d1_three_adaptations/{PREREGISTRATION,REPORT,STATUS,RESUME}.md and scripts/adaptation/{common,train_sft,train_residual,download_assets,prepare_base}.py.
- Visual alignment: `qrvmil/pi0-d0d1-visual-alignment@783bab5dc0502f4c7eccb4d7231a480a88dcc3d9`. Read README, REPORT_RU.md, docs/DESIGN_RU.md, configs/{pins,B-align,projector}.json and alignment/{model,build_cache,train}.py.
- OpenPI pin shared by both: `981483dca0fd9acba698fea00aa6e52d56a66c58`; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Native HF dataset `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`: all50 native demonstrations per task.
- D0 `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`; D1 `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- C-RADIO `nvidia/C-RADIOv3-L` revision `9d0413465e8a91e67bbf2c1ad342815478d1b906`.
- cuda_memtest `ComputationalRadiationPhysics/cuda_memtest@e94e1ee54e0689c9f154a8a08202554452552ec6`, compiled with CUDA12.8 for sm80.

## Historical checkpoints inspected, not new results

Exact HF revisions and per-file sizes/LFS hashes are in evidence/*-hf-inventory.json.

- Three-adaptation HF revision `57b978774e8e177d8d721b71f15542751e0610d9`: D0/3000, full/1500, expert/1500, teacher/250, residual final. Each full pi0 safetensors file is14,005,618,584 bytes. All old artifacts together occupy164.93GiB; do not download all.
- Alignment HF revision `7b37667b91dbc526c94b849b07d7ad370e8e0030`: B-control and B-align steps0/500/1000/2000/3001 plus frozen projector. Whole repo83.16GiB; do not download all.
- Original final results: full D0=0/50,D1=20/50; expert D0=0/50,D1=19/50; residual D0=0/50,D1=0/50. These are historical, not measurements from this instance.
- Alignment report: joint D0+D1 LoRA32 B-control versus B-align, 3001 updates; D0 n500=81.6% vs80.2%, D1 n50=38% vs60%. These cannot be compared causally to the sequential full-model recipes.

## Conditions to preserve from three-adaptation experiment

D0-only normalization, native obs[i]→action[i+1], rotate both cameras180°, official resize/pad/transforms, 50-action chunks, no duplicate inverse normalization. D0 full SFT3000 updates seed0; independent D1 full/expert1500 updates seed1 from identical finalD0; effective batch8, fp32 master weights/AdamW states, bf16 compute, gradient checkpointing. AdamW(.9,.95,eps1e-8,wd1e-10), clip1, cosine3e-5→3e-6, warmup100. Original microbatch2×accum4; only measured memory failure can justify changing that while preserving batch8. No LoRA.

Expert trains gemma_expert and state/action/time projections; entire PaliGemma/vision/projector remains frozen. Residual keeps pi0 fully frozen, uses shared SERL ResNet10 SAC/CalQL, offline1000 updates, teacher warmup20 episodes, active online50000 transitions, batch256, UTD1, offline fraction.5, actor/temp every2 critic steps, xi2, gamma.99,beta.1, potential shaping and terminalPhi0. Teacher is first fullD1 candidate250/500/750/1000/1250 with≥1/10 validation successes(seeds22000–22009); fail if absent.

Evaluation only after final weights fixed:300 rollouts=3methods×2tasks×50generated scenes, seeds30000–30049, horizon220,replan5, common persisted reset states/observation hashes and explicit common flow noise, deterministic residual actor; paired episode CSV/JSON, two-camera terminal-frame video, Wilson95% SR intervals, paired differences/exact McNemar with Holm across6comparisons. Validation/final/online seed sets remain disjoint.

## Alignment mechanism and unresolved scientific boundary

Transfer mechanism: frozen C-RADIO spatial teacher; PaliGemma block9(index8), only base/left-wrist visual prefix patches, exclude masked right-wrist; teacher14×14→student16×16 bilinear(align_corners=False); LN(eps1e-5)+MLP2048→1024→1024 with approximate GELU; calibration200updates, Adam1e-3,batch8, then frozen projector; cosine loss plus flow loss. K4 paired deterministic image augmentations require exactly the same RGB pixels for teacher and student. Pilot20updates with lambda0.2 and gradient-ratio rule from source, fresh weights for main run.

Do NOT silently import source joint D0+D1 training, D0+D1 normalization, LoRA32, JAX backend, 3001updates, LR2.5e-5/warmup1000 or seeds10000+. They change the requested comparison. Backend port and integration tolerance require explicit validation.

Critical: block9 prefix-only hidden features do not depend on action expert or residual actor. With the original frozen scopes, alignment loss gradient w.r.t. expert/residual parameters is exactly zero. Source pipeline rejects zero alignment gradient. Cannot label a constant auxiliary metric as effective alignment training. Options require an explicit binding before long training: align shared D0 representation then preserve expert/residual scopes; or change their trainable scope / align the SERL encoder and acknowledge a new method. D0-only versus joint projector calibration is also a data-exposure decision; do not silently expose D1 during a D0-only base.

## Storage and publication

New Git branch `exp/pi0-d0-d1-visual-alignment-20261029`; new HF repo (public by explicit user approval) `qrvmil/pi0-d0-d1-three-adaptations-visual-alignment-20261029`. Use explicit allowlists for evidence/config/code and newly created checkpoints only. Exclude credentials, datasets, teacher cache, upstream checkpoints, temporary files and unrelated historical results. Before each major stage require verified remote logs/checkpoint receipts. Long-training gate stays closed until VRAM and model/gradient/optimizer smoke pass and scientific binding is settled.
