from __future__ import absolute_import

import math
import unittest

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from clustercontrast.bridge_matching import (
    bridge_centers, bridge_selection_stats, ground_knn_scores,
    cross_retain_ratio, cross_view_mapping_confidences, gt_match_accuracy,
    matching_difference_stats, pclhd_bgm, random_bridge_scores,
    retain_mapping_by_confidence, select_near_members)


def _baseline_pgm_reference(cluster_features_rgb, cluster_features_ir):
    """Verbatim mapping logic from baseline engine's Stage-2 PGM block."""
    num_cluster_rgb = cluster_features_rgb.size(0)
    num_cluster_ir = cluster_features_ir.size(0)
    i2r = {}
    r2i = {}
    R = []
    bgm = False
    if num_cluster_rgb >= num_cluster_ir:
        # clusternorm
        cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
        cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
        similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T)) / 1).exp().cpu()
        dis_similarity = (1 / (similarity))
        cost = dis_similarity / 1
        tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
        cost = (torch.cat((cost, tmp), 1))
        unmatched_row = []
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, item in enumerate(row_ind):
            if col_ind[idx] < similarity.shape[1]:
                R.append((row_ind[idx], col_ind[idx]))
                r2i[row_ind[idx]] = col_ind[idx]
                i2r[col_ind[idx]] = row_ind[idx]
            else:
                unmatched_row.append(row_ind[idx])
        if bgm is False:
            unmatched_cost = cost[unmatched_row][:, :dis_similarity.shape[1]]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for idx, item in enumerate(unmatched_row_ind):
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
    return r2i, i2r


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

    def test_pclhd_bgm_matches_baseline_reference(self):
        generator = torch.Generator().manual_seed(23)
        # More ground clusters, the exact 2x fallback boundary, and fewer
        # ground clusters (where baseline disables cross-view loss).
        for ground_count, aerial_count in ((4, 3), (6, 3), (2, 3)):
            with self.subTest(
                    ground_count=ground_count, aerial_count=aerial_count):
                ground = torch.randn(
                    ground_count, 9, generator=generator)
                aerial = torch.randn(
                    aerial_count, 9, generator=generator)
                expected_r2i, expected_i2r = _baseline_pgm_reference(
                    ground.clone(), aerial.clone())
                actual_r2i, actual_i2r, stats = pclhd_bgm(
                    ground, aerial)
                self.assertEqual(actual_r2i, expected_r2i)
                self.assertEqual(actual_i2r, expected_i2r)
                self.assertEqual(stats["cross_view_enabled"], bool(expected_r2i))
        self.assertEqual(pclhd_bgm(ground[:2], aerial)[0], {})
        self.assertEqual(pclhd_bgm(ground[:2], aerial)[1], {})

    def test_ratio_one_degenerates_to_normalized_full_centers(self):
        generator = torch.Generator().manual_seed(29)
        features = F.normalize(
            torch.randn(9, 6, generator=generator), dim=1)
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        scores = torch.linspace(0.0, 1.0, len(labels))
        near, far = select_near_members(labels, scores, ratio=1.0)
        self.assertTrue(bool(near.all()))
        self.assertFalse(bool(far.any()))
        bridge = bridge_centers(features, labels, near)
        full = torch.stack([
            features[labels == cluster_id].mean(dim=0)
            for cluster_id in range(3)
        ])
        full = F.normalize(full, dim=1)
        self.assertTrue(torch.allclose(bridge, full, atol=1e-7))

        ground = torch.randn(5, 6, generator=generator)
        bridge_r2i, bridge_i2r, _ = pclhd_bgm(ground, bridge)
        full_r2i, full_i2r, _ = pclhd_bgm(ground, full)
        self.assertEqual(bridge_r2i, full_r2i)
        self.assertEqual(bridge_i2r, full_i2r)

    def test_cross_retain_schedule(self):
        self.assertAlmostEqual(cross_retain_ratio(0, 0.4, 1.0, 20), 0.4)
        self.assertAlmostEqual(cross_retain_ratio(10, 0.4, 1.0, 20), 0.7)
        self.assertAlmostEqual(cross_retain_ratio(20, 0.4, 1.0, 20), 1.0)
        self.assertAlmostEqual(cross_retain_ratio(99, 0.4, 1.0, 20), 1.0)

    def test_bidirectional_margin_and_similarity_confidence(self):
        ground = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        aerial = torch.tensor([[0.8, 0.2], [0.0, 1.0]])
        i2r = {0: 0, 1: 1}
        r2i = {0: 0, 1: 1, 2: 0}
        similarity = F.normalize(ground, dim=1).mm(
            F.normalize(aerial, dim=1).t())

        i_margin, r_margin = cross_view_mapping_confidences(
            ground, aerial, i2r, r2i, mode="margin")
        self.assertAlmostEqual(
            i_margin[0],
            float(similarity[0, 0] - similarity[1:, 0].max()), places=6)
        self.assertAlmostEqual(
            i_margin[1],
            float(similarity[1, 1]
                  - torch.stack((similarity[0, 1], similarity[2, 1])).max()),
            places=6)
        self.assertAlmostEqual(
            r_margin[0],
            float(similarity[0, 0] - similarity[0, 1]), places=6)

        i_similarity, r_similarity = cross_view_mapping_confidences(
            ground, aerial, i2r, r2i, mode="similarity")
        self.assertAlmostEqual(i_similarity[0], float(similarity[0, 0]))
        self.assertAlmostEqual(r_similarity[2], float(similarity[2, 0]))

    def test_retain_mapping_rounds_up_and_is_deterministic(self):
        mapping = {0: 10, 1: 11, 2: 12, 3: 13}
        confidence = {0: 0.2, 1: 0.9, 2: 0.8, 3: 0.1}
        retained, removed, stats = retain_mapping_by_confidence(
            mapping, confidence, ratio=0.4, total_clusters=4)
        self.assertEqual(retained, {1: 11, 2: 12})
        self.assertEqual(removed, {0: 10, 3: 13})
        self.assertEqual(stats["retained"], 2)
        self.assertAlmostEqual(stats["mean_confidence"], 0.85)

        retained_all, removed_all, _ = retain_mapping_by_confidence(
            {0: 10, 2: 12}, {0: 0.2, 2: 0.8},
            ratio=1.0, total_clusters=4)
        self.assertEqual(retained_all, {0: 10, 2: 12})
        self.assertEqual(removed_all, {})


if __name__ == "__main__":
    unittest.main()
