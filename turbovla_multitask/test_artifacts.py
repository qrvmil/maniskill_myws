import unittest
from data import TRAIN_SETS
from artifacts import require_final


class ArtifactContracts(unittest.TestCase):
    def test_smoke_checkpoint_cannot_be_published_as_final(self):
        row = dict(variant='A', step=3001, training_tasks=TRAIN_SETS['A'], smoke_only=True,
                   primary_weights='ema_model_state_dict')
        with self.assertRaisesRegex(ValueError,'smoke'):
            require_final(row,'A')

    def test_wrong_variant_or_partial_steps_rejected(self):
        row = dict(variant='B',step=3001,training_tasks=TRAIN_SETS['B'],smoke_only=False,
                   primary_weights='ema_model_state_dict')
        with self.assertRaises(ValueError):
            require_final(row,'A')
        row.update(variant='A',training_tasks=TRAIN_SETS['A'],step=500)
        with self.assertRaises(ValueError):
            require_final(row,'A')

    def test_final_ema_contract_accepted(self):
        row = dict(variant='C',step=3001,training_tasks=TRAIN_SETS['C'],smoke_only=False,
                   primary_weights='ema_model_state_dict')
        require_final(row,'C')


if __name__ == '__main__':
    unittest.main()
