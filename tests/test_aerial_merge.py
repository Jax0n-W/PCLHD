import hashlib
import json
import os.path as osp
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _train_agreid_bridge_engine as bridge_engine
import experiment_runner as runner
from clustercontrast.aerial_merge import (
    aerial_merge_association_only, fragment_groups, merge_aerial_clusters,
    merge_gt_stats, translate_group_mappings)
from scripts import run_all_ablations as batch_runner


def runner_snapshot(module, dataset, include_merge_modes=False):
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
        cases = [(experiment, [])]
        if include_merge_modes and module.EXPERIMENTS[experiment].get("bridge"):
            cases.extend([
                (experiment + "/threshold", ["--aerial-merge", "threshold"]),
                (experiment + "/adaptive", ["--aerial-merge", "adaptive"]),
            ])
        for name, extra in cases:
            args = module.build_parser().parse_args([
                "--dataset-key", dataset, "--experiment", experiment, "--dry-run"]
                + extra)
            results[name] = (
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


class AerialMergeAssociationTests(unittest.TestCase):
    def test_translate_expands_retained_groups_and_picks_nearest_fragment(self):
        group_of = fragment_groups([2, 1, -1, 0, 1], [1, 0, -1, 0, 0])
        self.assertEqual(group_of, {0: 0, 1: 0, 2: 1})
        aerial = torch.tensor([[10.0, 0.0], [0.0, 2.0], [0.0, -1.0]])
        ground = torch.zeros(6, 2)
        ground[5] = torch.tensor([1.0, 3.0])
        i2r, r2i = translate_group_mappings(
            group_of, {0: 5}, {5: 0}, aerial, ground)
        self.assertEqual(i2r, {0: 5, 1: 5})
        self.assertEqual(r2i, {5: 1})

    def test_translate_breaks_cosine_ties_by_smallest_fragment_id(self):
        aerial = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        ground = torch.tensor([[1.0, 0.0]])
        i2r, r2i = translate_group_mappings(
            {1: 0, 0: 0, 2: 1}, {}, {0: 0}, aerial, ground)
        self.assertEqual(i2r, {})
        self.assertEqual(r2i, {0: 0})

    def test_engine_validates_ratio_only_when_association_is_active(self):
        args = bridge_engine.build_parser().parse_args([
            "--stage1-checkpoint", "model_best.pth.tar",
            "--aerial-merge-scope", "association", "--bridge-ratio", "0.5"])
        bridge_engine._validate_args(args)
        self.assertFalse(aerial_merge_association_only(args))
        self.assertIs(bridge_engine._stage2_trainer_class(args),
                      bridge_engine.ClusterContrastTrainer_PCLMP_NoEMALoss)
        args.aerial_merge = "threshold"
        with self.assertRaisesRegex(ValueError, "--bridge-ratio 1.0"):
            bridge_engine._validate_args(args)
        args.bridge_ratio = 1.0
        bridge_engine._validate_args(args)
        self.assertIs(bridge_engine._stage2_trainer_class(args),
                      bridge_engine.ClusterContrastTrainer_PCLMP_ConfidentCross)

    def test_runner_reuses_previous_names_and_commands(self):
        # ac32c6f snapshots include off, threshold and adaptive label merging.
        for dataset, expected in {
                "agreid": "01fd6a796fbc2c41fa9cc0b6e2f988e806fa302eb4650e5ebd26d401a0cd5429",
                "agreidv2": "3756db57796ae41aa0b3901a6f71d6ac07ba40e3f73ce9355e1631bcde64ad8c",
        }.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(runner_snapshot(runner, dataset, True), expected)

    def test_runner_suffix_scope_and_effective_bridge_ratio(self):
        for mode, suffix in (("threshold", "_am0.7as"), ("adaptive", "_ama0.85as")):
            with self.subTest(mode=mode):
                args = runner.build_parser().parse_args([
                    "--experiment", "confident", "--aerial-merge", mode,
                    "--aerial-merge-lambda", "0.85",
                    "--aerial-merge-scope", "association", "--dry-run"])
                runner.validate_args(args)
                self.assertIn(suffix + "_channel_off", runner.default_log_names(args)[1])
                stage1 = runner.build_stage1_command(args)
                stage2 = runner.build_stage2_command(args, "model_best.pth.tar")
                self.assertNotIn("--aerial-merge-scope", stage1)
                self.assertEqual(stage2[stage2.index("--aerial-merge-scope") + 1],
                                 "association")
                engine_args = bridge_engine.build_parser().parse_args(stage2[2:])
                self.assertEqual(engine_args.bridge_ratio, 1.0)
                bridge_engine._validate_args(engine_args)
                args.experiment = "bridge"
                with self.assertRaisesRegex(ValueError, "--bridge-ratio 1.0"):
                    runner.validate_args(args)
                args.aerial_merge = "off"
                runner.validate_args(args)
                self.assertNotIn(suffix, runner.default_log_names(args)[1])
                self.assertNotIn("--aerial-merge-scope", runner.build_stage2_command(
                    args, "model_best.pth.tar"))

    def test_batch_forwards_association_scope_only_to_stage2(self):
        arguments = ["--experiments", "dynamic_confident", "--dry-run",
                     "--aerial-merge", "threshold",
                     "--aerial-merge-scope", "association"]
        with patch.object(batch_runner, "run") as run:
            batch_runner.main(arguments)
        stage1, stage2 = (call.args[0] for call in run.call_args_list)
        self.assertNotIn("--aerial-merge-scope", stage1)
        self.assertEqual(stage2[stage2.index("--aerial-merge-scope") + 1],
                         "association")
        arguments[1] = "dynamic_bridge"
        with patch.object(batch_runner, "run") as run:
            with self.assertRaisesRegex(ValueError, "--bridge-ratio 1.0"):
                batch_runner.main(arguments)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
