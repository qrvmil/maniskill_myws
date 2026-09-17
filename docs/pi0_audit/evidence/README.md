# Audit evidence

The [main report](../../PI0_D0_D1_SANITY_REPORT.md) explains the scientific
results. This directory retains the smaller inputs needed to inspect how those
results were obtained. Large model weights and demonstration datasets are not
committed.

- `setup/`: hardware and package inventory, official checkpoint verification,
  preregistration timing, raw-checkpoint normalization inspection, and review
  notes. `setup/supervisor/` preserves the three job wrappers and service configs.
- `run/`: D0 dataset/statistics binding, training configuration, actual optimizer
  update records, original evaluation rows and summaries, prompt evidence,
  checkpoint bindings and parameter manifests, and first-action arrays.
- `runtime/`: stage metadata, package versions and sampled device memory.
- `source_snapshots.tar.gz`: the source/config snapshots captured for each stage.
  Its directory names identify the corresponding stage. The official control's
  later audit-entrypoint capture is explicitly labeled.
- `verification/`: tests, video inspection records, and final consistency checks.

Paths inside original JSON files retain their run-time absolute values. They are
provenance, not a claim that the large external files are stored in Git. The
`checkpoint_params_sha256.json` files record checkpoint bytes at evaluation
load time; `update_*.json` records verify completed optimizer updates at save
time. Final report generation cross-checks the model, path, update record,
normalization binding, registered seeds and parameter-manifest digest.

The training metadata's unused `model: positive_control` CLI default did not
select the training model: `run/training_config.txt` records the D0-only pi0
configuration and official pi0 base weight loader. See the
[methods explanation](../METHODS.md#runtime-provenance).

Reproducing the run requires the pinned OpenPI/LIBERO environment and external
assets described in the methods and preregistration. The audit entrypoint has
`bind`, `train`, and `eval --model MODEL` modes. The serial pipeline runs primary
step 0, training, auxiliary LoRA initialization and the four learned milestones
after the positive control finishes. No evaluation mode trains a model.
The wrappers are specific to this Vast instance and source its environment;
the run used `HF_LEROBOT_HOME=/workspace/audit-data/lerobot` in addition to the
CUDA/EGL settings visible in those wrappers. They do not expose a web service.

After all original evaluations and selected video replays complete, the final
report command is:

```bash
scripts/pld/libero_python.sh scripts/pld/write_pi0_audit_report.py
```

It rebuilds tables, action diagnostics and figures, rechecks all publication
videos against original episode rows, and requires inspected visible-behavior
notes before writing the report and video index. The source of each figure and
its final file checksum are recorded in `../figure_provenance.json`.

The figures and numerical tables can also be rebuilt from this committed evidence
without the external model weights or demonstration files:

```bash
scripts/pld/libero_python.sh scripts/pld/report_pi0_audit.py \
  --work docs/pi0_audit/evidence/run --output /tmp/pi0_audit_rebuilt
```

This uses the archived optimizer-update records to validate the original model
paths. Relocating the evidence does not change its recorded checkpoint identity.

The full relevant verification command is:

```bash
PLD_LIBERO_INTEGRATION=1 scripts/pld/libero_python.sh -m pytest -q
```

Deterministic video replays are stored separately from original episode rows.
They do not add to the 700 original evaluation episodes. The two earlier,
pre-addendum D1 recordings are retained and verified separately; the published
video index uses the canonical filenames and includes terminal observations.

Final validation is recorded in [verification/final_checks.json](verification/final_checks.json). The offline rebuild matched all nine numerical/PNG outputs byte for byte; PDFs were parsed separately. The [test log](verification/final-tests.log) records 126 passed and 4 skipped with real MuJoCo integration enabled. The [review record](verification/review_notes.md) summarizes independent numerical, figure, report and video checks.
