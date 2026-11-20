from __future__ import absolute_import

import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from clustercontrast.bridge_matching import (
    bridge_centers, bridge_selection_stats, ground_knn_scores,
    gt_match_accuracy, matching_difference_stats, random_bridge_scores,
    select_near_members)


class BridgeMatchingTests(unittest.TestCase):

    def test_cluster_selection_rounds_up_keeps_one_and_excludes_noise(self):
        labels = torch.tensor([0, 0, 0, 1, -1])
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 1.0])
        near, far = select_near_members(labels, scores, ratio=0.5)
        self.assertEqual(near.tolist(), [False, True, True, True, False])
        self.assertEqual(far.tolist(), [True, False, False, False, False])
        stats = bridge_selection_stats(labels, scores, near)
        self.assertEqual(stats["aerial_clusters"], 2)
        self.assertEqual(stats["near_images"], 3)
        self.assertEqual(stats["far_images"], 1)
        self.assertEqual(stats["outliers"], 1)

    def test_random_mode_is_reproducible_by_seed_and_epoch(self):
        first = random_bridge_scores(20, seed=7, epoch=3)
        second = random_bridge_scores(20, seed=7, epoch=3)
        changed = random_bridge_scores(20, seed=7, epoch=4)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, changed))

    def test_bridge_centers_are_normalized_means_of_normalized_members(self):
        features = torch.tensor([
            [3.0, 0.0], [0.0, 4.0], [2.0, 0.0], [0.0, 5.0]])
        labels = torch.tensor([0, 0, 1, 1])
        near = torch.tensor([True, True, False, True])
        actual = bridge_centers(features, labels, near)
        expected = torch.stack([
            F.normalize(torch.tensor([0.5, 0.5]), dim=0),
            torch.tensor([0.0, 1.0]),
        ])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_ground_knn_chunked_matches_full_matrix(self):
        generator = torch.Generator().manual_seed(11)
        aerial = torch.randn(13, 7, generator=generator)
        ground = torch.randn(9, 7, generator=generator)
        expected = ground_knn_scores(aerial, ground, topk=4, chunk_size=13)
        actual = ground_knn_scores(aerial, ground, topk=4, chunk_size=3)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_gt_match_accuracy_uses_cluster_majority_identity(self):
        aerial_labels = np.array([0, 0, 0, 1, 1])
        aerial_names = [
            "P0001T00001A0C0F1.jpg",
            "P0001T00001A0C0F2.jpg",
            "P0002T00002A0C0F3.jpg",
            "P0003T00003A0C0F4.jpg",
            "P0003T00003A0C0F5.jpg",
        ]
        ground_labels = np.array([0, 0, 1, 1])
        ground_names = [
            "P0001T00001A0C2F1.jpg",
            "P0001T00001A0C2F2.jpg",
            "P0004T00004A0C3F3.jpg",
            "P0004T00004A0C3F4.jpg",
        ]
        result = gt_match_accuracy(
            aerial_labels, aerial_names, ground_labels, ground_names,
            {0: 0, 1: 1})
        self.assertEqual(result["correct_pairs"], 1)
        self.assertEqual(result["evaluated_pairs"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.5)

    def test_matching_difference_ratio(self):
        result = matching_difference_stats(
            {0: 1, 1: 0, 2: 2}, {0: 1, 1: 2, 2: 2})
        self.assertEqual(result["different_clusters"], 1)
        self.assertAlmostEqual(result["different_ratio"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
