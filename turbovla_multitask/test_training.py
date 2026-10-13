import unittest
import numpy as np
import torch
import training
import tempfile
from pathlib import Path


class TrainingContracts(unittest.TestCase):
    def test_live_training_lock_rejects_another_writer_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with training.training_lock(directory):
                with self.assertRaises(BlockingIOError):
                    with training.training_lock(directory): pass
            with training.training_lock(directory): pass

    def test_padding_has_no_loss_or_gradient(self):
        prediction = torch.tensor([[[1.]*7, [100.]*7]], requires_grad=True)
        loss = training.masked_l1(prediction, torch.zeros_like(prediction), torch.tensor([[1., 0.]]))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[:,1], torch.zeros(1,7)))

    def test_ema_uses_predeclared_decay(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.)
        ema = {'weight':torch.tensor([[0.]])}
        training.update_ema(ema, model, .999)
        self.assertAlmostEqual(ema['weight'].item(), .002, places=7)

    def test_task_sampling_is_uniform_and_reproducible(self):
        lengths = [10, 1000, 20]
        first = training.sample_indices(lengths, step=42, batch_size=6000, seed=0)
        self.assertEqual(first, training.sample_indices(lengths,42,6000,0))
        counts = np.bincount([task for task,frame in first],minlength=3)
        self.assertTrue((np.abs(counts-2000) < 150).all(),counts)
        self.assertTrue(all(0 <= frame < lengths[task] for task,frame in first))


if __name__ == '__main__':
    unittest.main()
