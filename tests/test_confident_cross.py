from __future__ import absolute_import

import unittest

import torch

from _train_agreid_bridge_engine import _validate_args, build_parser
from clustercontrast.bridge_matching import (
    cross_view_mapping_confidences, retain_mapping_by_confidence)
from clustercontrast.three_domain import build_bidirectional_bgm
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


class ConfidentCrossMatchingTests(unittest.TestCase):

    def test_nearest_is_valid_with_full_cluster_centers(self):
        args = build_parser().parse_args([
            "--stage1-checkpoint", "model_best.pth.tar",
            "--confident-cross", "--bgm-fallback", "nearest",
            "--bridge-ratio", "1.0"])
        _validate_args(args)
        args.bridge_ratio = 0.5
        with self.assertRaisesRegex(ValueError, "--bridge-ratio 1.0"):
            _validate_args(args)

    def test_nearest_scores_and_filters_both_sides_of_rectangular_matching(self):
        generator = torch.Generator().manual_seed(42)
        ground = torch.randn(3, 8, generator=generator)
        aerial = torch.randn(5, 8, generator=generator)
        r2i, i2r, stats = build_bidirectional_bgm(ground, aerial)
        self.assertEqual(set(i2r), set(range(5)))
        self.assertEqual(set(r2i), set(range(3)))
        self.assertEqual(stats["hungarian_pairs"], 3)
        self.assertEqual(stats["target_fallbacks"], 2)

        i2r_scores, r2i_scores = cross_view_mapping_confidences(
            ground, aerial, i2r, r2i, mode="margin")
        self.assertEqual(set(i2r_scores), set(i2r))
        self.assertEqual(set(r2i_scores), set(r2i))
        for aerial_id, ground_id in i2r.items():
            if r2i[ground_id] != aerial_id:
                self.assertGreaterEqual(i2r_scores[aerial_id], 0.0)

        for mapping, scores, total in ((i2r, i2r_scores, 5),
                                       (r2i, r2i_scores, 3)):
            retained, removed, _ = retain_mapping_by_confidence(
                mapping, scores, 0.4, total)
            self.assertEqual(len(retained), 2)
            self.assertEqual(len(removed), total - 2)
            self.assertEqual({**retained, **removed}, mapping)
            retained, removed, _ = retain_mapping_by_confidence(
                mapping, scores, 1.0, total)
            self.assertEqual(retained, mapping)
            self.assertEqual(removed, {})


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
