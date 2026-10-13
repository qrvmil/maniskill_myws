import unittest
import numpy as np
import data


class DataContracts(unittest.TestCase):
    def test_terminal_chunk_repeats_last_action_and_masks_padding(self):
        actions = np.arange(21, dtype=np.float32).reshape(3, 7)
        chunk, mask = data.action_chunk(actions, 1, 4)
        np.testing.assert_array_equal(chunk, actions[[1, 2, 2, 2]])
        np.testing.assert_array_equal(mask, [1, 1, 0, 0])

    def test_normalization_round_trip_preserves_binary_gripper(self):
        actions = np.array([[-.5]*6+[-1], [.8]*6+[1]], np.float32)
        states = np.arange(16, dtype=np.float32).reshape(2, 8)
        stats = data.compute_stats([states], [actions])
        normalized = data.normalize_actions(actions, stats)
        np.testing.assert_allclose(normalized[:, :6], [[-1]*6, [1]*6], atol=2e-6)
        np.testing.assert_array_equal(normalized[:, 6], [-1, 1])
        np.testing.assert_allclose(data.denormalize_actions(normalized, stats), actions, atol=2e-6)

    def test_heldout_task_rejected_from_training(self):
        with self.assertRaisesRegex(ValueError, 'training'):
            data.training_tasks(['D0', 'H1'])

    def test_zero_variance_stats_remain_finite(self):
        stats = data.compute_stats([np.ones((2, 8))], [np.ones((2, 7))])
        self.assertTrue(np.isfinite(data.normalize_actions(np.ones((2, 7)), stats)).all())

    def test_unknown_instruction_is_not_added_to_training(self):
        self.assertIsNone(data.task_for_instruction('pick up a held out object'))
        self.assertEqual(data.task_for_instruction('put the bowl on the stove'), 'D2')


if __name__ == '__main__':
    unittest.main()
