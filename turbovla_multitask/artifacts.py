"""Require a verified full experiment checkpoint before evaluation/publication."""
import json
from data import WORK, TRAIN_SETS
from prepare import sha256


def require_final(metadata, variant):
    if metadata.get('smoke_only') is not False:
        raise ValueError('A smoke/unverified checkpoint cannot be used as final')
    if metadata.get('variant') != variant or metadata.get('training_tasks') != TRAIN_SETS[variant]:
        raise ValueError('Final checkpoint belongs to a different train set')
    if metadata.get('step') != 3001:
        raise ValueError('Final checkpoint has not completed 3001 updates')
    if metadata.get('primary_weights') != 'ema_model_state_dict':
        raise ValueError('Primary weights must be the predeclared EMA')


def verified_final(variant):
    run = WORK/variant/'train'
    metadata = json.loads((run/'complete.json').read_text())
    require_final(metadata,variant)
    checkpoint = run/'final.pth'
    stats = WORK/variant/'stats.json'
    if sha256(checkpoint) != metadata['checkpoint_sha256'] or sha256(stats) != metadata['stats_sha256']:
        raise ValueError('Final checkpoint or statistics changed')
    return checkpoint,stats,metadata
