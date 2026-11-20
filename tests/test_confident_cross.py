from __future__ import absolute_import

import unittest

import torch

from clustercontrast.trainers_confident_cross import (
    ClusterContrastTrainer_PCLMP_ConfidentCross,
    confident_cross_view_loss, mapped_memory_loss)
from clustercontrast.trainers_no_ema_loss import (
    ClusterContrastTrainer_PCLMP_NoEMALoss,
    compose_primary_loss_without_ema)


class _MeanMemory(object):

    def __call__(self, features, targets, model_name="encoder"):
        per_sample = features.pow(2).mean(dim=1) + targets.float() * 0.1
        return per_sample.mean()


class ConfidentCrossTrainerTests(unittest.TestCase):

    def test_trainer_inherits_no_ema_loss_trainer(self):
        self.assertTrue(issubclass(
            ClusterContrastTrainer_PCLMP_ConfidentCross,
            ClusterContrastTrainer_PCLMP_NoEMALoss))

    def test_rho_one_cross_and_total_losses_equal_baseline(self):
        memory_ir = _MeanMemory()
        memory_rgb = _MeanMemory()
        features_rgb = torch.tensor([
            [1.0, 2.0], [3.0, 1.0], [0.5, 2.5], [1.5, 0.5]])
        features_ir = torch.tensor([
            [0.2, 1.2], [2.2, 0.4], [1.1, 1.4]])
        labels_rgb = torch.tensor([0, 1, 2, 1])
        labels_ir = torch.tensor([0, 1, 2])
        r2i = {0: 2, 1: 0, 2: 1}
        i2r = {0: 1, 1: 2, 2: 0}

        expected_even = memory_ir(
            features_rgb,
            torch.tensor([r2i[int(label)] for label in labels_rgb]))
        actual_even = confident_cross_view_loss(
            0, memory_ir, memory_rgb, features_rgb, features_ir,
            labels_rgb, labels_ir, i2r, r2i)
        self.assertTrue(torch.equal(actual_even, expected_even))

        expected_odd = memory_rgb(
            features_ir,
            torch.tensor([i2r[int(label)] for label in labels_ir]))
        actual_odd = confident_cross_view_loss(
            1, memory_ir, memory_rgb, features_rgb, features_ir,
            labels_rgb, labels_ir, i2r, r2i)
        self.assertTrue(torch.equal(actual_odd, expected_odd))

        loss_ir = torch.tensor(1.25)
        loss_rgb = torch.tensor(0.75)
        baseline_total = compose_primary_loss_without_ema(
            loss_ir, loss_rgb, expected_even)
        confident_total = compose_primary_loss_without_ema(
            loss_ir, loss_rgb, actual_even)
        self.assertTrue(torch.equal(confident_total, baseline_total))

    def test_partial_mapping_averages_only_valid_samples(self):
        memory = _MeanMemory()
        features = torch.tensor([
            [1.0, 0.0], [3.0, 1.0], [2.0, 2.0], [4.0, 0.0]])
        labels = torch.tensor([0, 1, 2, 1])
        mapping = {1: 4}
        actual = mapped_memory_loss(memory, features, labels, mapping)
        expected = memory(features[[1, 3]], torch.tensor([4, 4]))
        self.assertTrue(torch.equal(actual, expected))

    def test_empty_mapping_returns_zero(self):
        memory = _MeanMemory()
        features = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        loss = mapped_memory_loss(memory, features, labels, {})
        self.assertEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
