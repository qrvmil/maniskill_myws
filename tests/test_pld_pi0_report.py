"""Reject mislabeled checkpoint evidence before constructing scientific tables."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


def test_report_binding_rejects_wrong_milestone_or_record(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'audit_report', Path('scripts/pld/report_pi0_audit.py'))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    checkpoint = tmp_path / 'sft/checkpoints/pi0_libero_seen_lora32/EXP-001/500'
    record = dict(optimizer_updates=500, upstream_loop_index=499,
                  checkpoint_directory=str(checkpoint), normalization_sha256='frozen')
    (tmp_path / 'sft').mkdir()
    (tmp_path / 'sft/update_0.json').write_text(json.dumps(dict(
        optimizer_updates=0, upstream_loop_index=None,
        checkpoint_directory=str(checkpoint.with_name('0')), normalization_sha256='frozen')))
    record_path = tmp_path / 'sft/update_500.json'
    record_path.write_text(json.dumps(record))
    binding = dict(model='500', checkpoint=str(checkpoint), normalization_sha256='frozen',
                   seed_block=list(range(9000, 9050)), action_horizon=50, action_dim=32)
    report.validate_model_identity(tmp_path, '500', binding)
    for change in [dict(model='1000'), dict(checkpoint=str(checkpoint.with_name('1000'))),
                   dict(seed_block=list(range(8000, 8050))), dict(action_horizon=10)]:
        with pytest.raises(ValueError):
            report.validate_model_identity(tmp_path, '500', dict(binding, **change))
    for change in [dict(optimizer_updates=499), dict(upstream_loop_index=500),
                   dict(checkpoint_directory=str(checkpoint.with_name('1000'))),
                   dict(normalization_sha256='other')]:
        record_path.write_text(json.dumps(dict(record, **change)))
        with pytest.raises(ValueError):
            report.validate_model_identity(tmp_path, '500', binding)
    record_path.write_text(json.dumps(record))
    archive = tmp_path / 'archived_evidence'
    shutil.copytree(tmp_path / 'sft', archive / 'sft')
    # Archived metadata keeps the original absolute run paths, without weights.
    report.validate_model_identity(archive, '500', binding)
