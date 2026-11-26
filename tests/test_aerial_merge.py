import hashlib
import json
import os.path as osp
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import experiment_runner as runner
from clustercontrast.aerial_merge import merge_aerial_clusters, merge_gt_stats
from scripts import run_all_ablations as batch_runner


def runner_snapshot(module, dataset):
    """Hash names and commands with machine-specific paths normalized."""
    path_flags = {
        "--data-dir", "--logs-dir", "--dynamic-agva-mask-dir",
        "--dynamic-agva-image-root", "--view-score-file", "--stage1-checkpoint",
    }

    def normalize(command):
        return ["<PYTHON>", osp.basename(command[1])] + [
            "<PATH>" if command[i - 1] in path_flags else value
            for i, value in enumerate(command[2:], start=2)]

    results = {}
    for experiment in sorted(module.EXPERIMENTS):
        args = module.build_parser().parse_args([
            "--dataset-key", dataset, "--experiment", experiment, "--dry-run"])
        results[experiment] = (
            module.default_log_names(args),
            normalize(module.build_stage1_command(args)),
            normalize(module.build_stage2_command(args, "model_best.pth.tar")),
        )
    return hashlib.sha256(json.dumps(
        results, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class AerialMergeTests(unittest.TestCase):
    def test_threshold_consolidates_one_to_three_fragments_per_identity(self):
        # Three identities, with their minimum old labels in different order
        # from the input rows. The final feature is an outlier.
        features = torch.tensor([
            [1.0, 0.01, 0.0], [1.0, 0.03, 0.0], [1.0, -0.02, 0.0],
            [0.0, 1.0, 0.0], [-0.01, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
        ])
        labels = np.array([12, 8, 14, 3, 9, 20, -1])
        merged, num_clusters, stats = merge_aerial_clusters(
            features, labels, "threshold")
        np.testing.assert_array_equal(merged, [1, 1, 1, 0, 0, 2, -1])
        np.testing.assert_array_equal(labels, [12, 8, 14, 3, 9, 20, -1])
        self.assertEqual(num_clusters, 3)
        self.assertEqual(stats, {
            "clusters_before": 6, "clusters_after": 3, "merges": 3,
            "merged_groups": 2, "largest_group": 3,
        })
        order = [6, 3, 5, 1, 0, 2, 4]
        shuffled, _, _ = merge_aerial_clusters(
            features[order], labels[order], "threshold")
        np.testing.assert_array_equal(shuffled, merged[order])

    def test_off_preserves_the_original_labels_and_ignores_features(self):
        labels = np.array([5, 2, 5, -1])
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        result, num_clusters, stats = merge_aerial_clusters(None, labels, "off")
        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[:1] + numpy_after[2:],
                         numpy_rng[:1] + numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertIs(result, labels)
        np.testing.assert_array_equal(result, [5, 2, 5, -1])
        self.assertEqual(num_clusters, 2)
        self.assertEqual(stats["merges"], 0)

    def test_single_link_merges_nonadjacent_members_of_the_chain(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        merged, num_clusters, stats = merge_aerial_clusters(
            features, [0, 1, 2], "threshold", tau=0.7)
        np.testing.assert_array_equal(merged, [0, 0, 0])
        self.assertEqual(num_clusters, 1)
        self.assertEqual(stats["largest_group"], 3)

    def test_adaptive_compactness_and_lambda(self):
        # The first cluster's center is (1, 0), compactness is 0.8.
        # The second is a singleton, so its compactness is 1.0.
        # Their cosine is 0.75, between the cutoffs 0.72 and 0.81.
        features = torch.tensor([[8.0, 6.0], [8.0, -6.0],
                                 [0.75, (1 - 0.75 ** 2) ** 0.5]])
        for lam, expected_labels, expected_merges in (
                (0.8, [0, 0, 0], 1), (0.9, [0, 0, 1], 0)):
            with self.subTest(lam=lam):
                merged, _, stats = merge_aerial_clusters(
                    features, [0, 0, 1], "adaptive", lam=lam)
                np.testing.assert_array_equal(merged, expected_labels)
                self.assertAlmostEqual(stats["median_compactness"], 0.9)
                self.assertEqual(stats["merges"], expected_merges)

    def test_gt_pair_precision_and_sample_weighted_bcubed(self):
        identities = [1, 1, 1, 2, 2, 3]
        names = ["P{:04d}T00001A0C0F{}.jpg".format(pid, index)
                 for index, pid in enumerate(identities)] + ["ignored.jpg"]
        stats = merge_gt_stats(
            [8, 8, 3, 11, 11, 20, -1], [0, 0, 0, 0, 0, 1, -1], names)
        self.assertEqual(stats["merged_pairs"], 3)
        self.assertAlmostEqual(stats["pair_precision"], 1 / 3)
        for side, expected in (
                ("bcubed_before", (1.0, 7 / 9, 7 / 8)),
                ("bcubed_after", (3 / 5, 1.0, 3 / 4))):
            for key, value in zip(("precision", "recall", "f"), expected):
                self.assertAlmostEqual(stats[side][key], value)


class AerialMergeRunnerTests(unittest.TestCase):
    def test_default_names_and_commands_match_3fabd47(self):
        # Snapshots taken from 3fabd47 for all eight experiments.
        for dataset, expected in {
                "agreid": "d86242331804f47ca74bcb0ff6b59c255bbd716d7f737e83ae49c610429e6e67",
                "agreidv2": "cc5414ebe902aa11257e295c3079ec475f795e971acbd055c2a585241b24839c",
        }.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(runner_snapshot(runner, dataset), expected)

    def test_nearest_and_threshold_suffixes_and_stage2_arguments(self):
        args = runner.build_parser().parse_args([
            "--experiment", "confident", "--bgm-fallback", "nearest",
            "--aerial-merge", "threshold", "--eps", "0.6", "--dry-run"])
        runner.validate_args(args)
        self.assertEqual(runner.default_log_names(args), (
            "agreid_dt_s1_d0_channel_off_seed1_eps0.6",
            "agreid_dt_confident_d0_cr0.4-1.0r20_nnfb_am0.7_channel_off_s2_seed1_eps0.6",
        ))
        stage1 = runner.build_stage1_command(args)
        stage2 = runner.build_stage2_command(args, "model_best.pth.tar")
        for flag, expected in (
                ("--aerial-merge", "threshold"), ("--aerial-merge-tau", "0.7"),
                ("--aerial-merge-lambda", "0.9")):
            self.assertNotIn(flag, stage1)
            self.assertEqual(stage2[stage2.index(flag) + 1], expected)

    def test_adaptive_suffix_and_batch_stage2_only_passthrough(self):
        arguments = ["--aerial-merge", "adaptive", "--aerial-merge-lambda", "0.85"]
        args = runner.build_parser().parse_args([
            "--experiment", "dynamic_bridge", "--dry-run"] + arguments)
        self.assertIn("_b1_ama0.85_channel_off", runner.default_log_names(args)[1])
        with patch.object(batch_runner, "run") as run:
            batch_runner.main([
                "--experiments", "dynamic_bridge", "confident", "--dry-run"]
                + arguments)
        commands_by_stage = {"stage1": [], "stage2": []}
        for call in run.call_args_list:
            command = call.args[0]
            stage = command[command.index("--run-mode") + 1]
            commands_by_stage[stage].append(command)
        stage1 = commands_by_stage["stage1"]
        stage2 = commands_by_stage["stage2"]
        self.assertEqual(len(stage1), 2)
        self.assertEqual(len(stage2), 2)
        for command in stage1:
            self.assertNotIn("--aerial-merge", command)
        for command in stage2:
            for flag, value in zip(arguments[::2], arguments[1::2]):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_non_bridge_experiments_fail_before_launch(self):
        for experiment, spec in runner.EXPERIMENTS.items():
            if not spec.get("bridge", False):
                with self.subTest(experiment=experiment):
                    args = runner.build_parser().parse_args([
                        "--experiment", experiment, "--aerial-merge", "threshold",
                        "--dry-run"])
                    with self.assertRaisesRegex(ValueError, "--experiment bridge"):
                        runner.validate_args(args)
        with patch.object(batch_runner, "run") as run:
            with self.assertRaisesRegex(ValueError, "--experiment bridge"):
                batch_runner.main([
                    "--experiments", "bridge", "baseline",
                    "--aerial-merge", "threshold", "--dry-run"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
