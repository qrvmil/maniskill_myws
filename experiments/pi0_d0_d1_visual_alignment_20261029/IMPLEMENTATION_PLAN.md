# Visual alignment reproduction implementation plan

Goal: reproduce three sequential adaptations with an aligned common D0 base under the approved scopes and unchanged data/update/evaluation budgets.
Architecture: preserve pinned PyTorch OpenPI/LIBERO and original adaptation implementation; add deterministic teacher-cache input replay, prefix-only block9 features, frozen projector cosine loss, durable publication barriers, and separate artifact destinations.
Spec: PREREGISTRATION.md and SOURCE_AUDIT.md.

- [x] Read both sources/reports/configs and inventory current HF checkpoints at immutable revisions.
- [x] Verify host/GPU/CPU/driver/toolkit/disk and actual local GitHub/HF identities, write/read probe and initial remote audit.
- [x] Bind common aligned D0 and original scopes with the user.
- [ ] Complete uncapped VRAM pattern/address/stress test; check allocation coverage, return code, ECC and temperatures. Any failure forbids long training.
- [ ] Install pinned runtime; verify PyTorch/JAX GPU operations.
- [ ] Verify projector/cosine numerical port, masks and gradient flow; verify block9 prefix against full model.
- [ ] Run discarded actual pretrained-model/teacher/alignment/backward/AdamW smoke with effective batch8 and sufficient measured headroom. Save logs remotely.
- [ ] Only after all gates pass, implement main cache/calibration/pilot/training and resumable publisher using separate paths, preserving original algorithms.
- [ ] Verify original D0 normalization byte hash, input revisions, all50demos/task and the full/expert sample-order identity.
- [ ] Run D0calibration200,pilot20,freshD0main3000 with progressive durable saves; validate gates before D1.
- [ ] Run fullD1/expertD1 1500 each, validate early teacher, frozen tensor hashes, then original residual1000offline+20warmup+50000active.
- [ ] Publish and verify all final checkpoint bindings before opening final seeds. Run identical300episodes and build concise table/report with verified artifact links.

Review focus: frozen scopes yielding zero alignment gradient; student/teacher pixel mismatch from extra preprocessing; memory occupied by teacher/JAX pools during policy update; inherited old completion markers or hardcoded HF destinations; incomplete publication/deletion of optimizer snapshots.
