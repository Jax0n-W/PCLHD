from __future__ import absolute_import

import argparse
import os.path as osp
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_color
from clustercontrast.utils.data.resolution_augment import (
    CrossDomainResolutionAugment, build_height_distribution,
    build_resolution_augmenters, sample_non_overlapping_rectangles)
from clustercontrast.utils.resolution_buckets import (
    evaluate_resolution_buckets)
from examples.PCLHD_AGReID_g2ag import eval_agreid


def _image(width=40, height=80):
    pixels = np.arange(width * height * 3, dtype=np.uint8).reshape(
        height, width, 3)
    return Image.fromarray(pixels, "RGB")


def _args(**updates):
    values = {
        "res_aug_mode": "global",
        "res_aug_prob": 1.0,
        "res_aug_domains": "auto",
        "res_aug_stages": "both",
        "res_aug_min_scale": 0.15,
        "res_aug_patches": (3, 5),
        "res_aug_patch_area": (0.05, 0.20),
    }
    values.update(updates)
    return argparse.Namespace(**values)


class _ArrayTransform(object):
    def __call__(self, image):
        return np.asarray(image).copy()


class ResolutionAugmentCoreTests(unittest.TestCase):
    def test_scale_at_least_one_is_exactly_unchanged(self):
        image = _image(height=80)
        augmenter = CrossDomainResolutionAugment(
            (80, 120), probability=1.0, mode="global")
        output = augmenter(image, rng=random.Random(2))
        self.assertIs(output, image)
        self.assertEqual(output.tobytes(), image.tobytes())

    def test_downscale_restores_original_size(self):
        image = _image(height=80)
        augmenter = CrossDomainResolutionAugment(
            (20,), probability=1.0, mode="global")
        output = augmenter(image, rng=random.Random(3))
        self.assertEqual(output.size, image.size)
        self.assertNotEqual(output.tobytes(), image.tobytes())

    def test_multi_rectangles_are_in_bounds_nonoverlapping_and_in_range(self):
        rectangles, requested, fallbacks = \
            sample_non_overlapping_rectangles(
                (144, 288), (3, 5), (0.05, 0.10),
                rng=random.Random(11))
        self.assertGreaterEqual(requested, 3)
        self.assertLessEqual(requested, 5)
        self.assertEqual(len(rectangles), requested)
        self.assertEqual(fallbacks, 0)
        for left, top, right, bottom in rectangles:
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, 144)
            self.assertLessEqual(bottom, 288)
            self.assertLess(left, right)
            self.assertLess(top, bottom)
        for index, first in enumerate(rectangles):
            for second in rectangles[index + 1:]:
                overlap_width = min(first[2], second[2]) - max(
                    first[0], second[0])
                overlap_height = min(first[3], second[3]) - max(
                    first[1], second[1])
                self.assertTrue(overlap_width <= 0 or overlap_height <= 0)

    def test_multi_falls_back_when_patches_do_not_fit(self):
        rectangles, requested, fallbacks = \
            sample_non_overlapping_rectangles(
                (20, 20), (4, 4), (0.90, 0.95),
                rng=random.Random(5), max_attempts=10)
        self.assertEqual(requested, 4)
        self.assertLess(len(rectangles), requested)
        self.assertEqual(fallbacks, requested - len(rectangles))

    def test_same_seed_reproduces_pixels_and_patch_stats(self):
        image = _image()
        outputs = []
        snapshots = []
        for _ in range(2):
            augmenter = CrossDomainResolutionAugment(
                (18, 25, 36), probability=1.0, mode="multi")
            outputs.append(augmenter(image, rng=random.Random(1234)))
            snapshots.append(augmenter.snapshot())
        self.assertEqual(outputs[0].tobytes(), outputs[1].tobytes())
        self.assertEqual(snapshots[0], snapshots[1])

    def test_same_seed_reproduces_one_worker_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = osp.join(directory, "sample.png")
            _image().save(path)
            records = [(path, index, 0) for index in range(4)]

            def collect(seed):
                random.seed(seed)
                torch.manual_seed(seed)
                augmenter = CrossDomainResolutionAugment(
                    (18, 25, 36), probability=1.0, mode="multi")
                loader = DataLoader(
                    Preprocessor(
                        records, transform=_ArrayTransform(),
                        resolution_process=augmenter),
                    batch_size=2, shuffle=False, num_workers=1)
                batches = [batch[0].numpy().copy() for batch in loader]
                return batches, augmenter.snapshot()

            first_batches, first_stats = collect(77)
            second_batches, second_stats = collect(77)
            self.assertEqual(first_stats, second_stats)
            self.assertEqual(len(first_batches), len(second_batches))
            for first, second in zip(first_batches, second_batches):
                np.testing.assert_array_equal(first, second)

    def test_height_distribution_uses_only_supplied_train_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            test = root / "test"
            train.mkdir()
            test.mkdir()
            Image.new("RGB", (10, 40)).save(train / "a.jpg")
            Image.new("RGB", (10, 60)).save(train / "b.jpg")
            Image.new("RGB", (10, 999)).save(test / "leak.jpg")
            records = [(str(train / "a.jpg"), 0, 0),
                       (str(train / "b.jpg"), 1, 0)]
            self.assertEqual(build_height_distribution(records), (40, 60))

    def test_cross_domain_builder_never_reads_unsupplied_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, height in (
                    ("aerial_train", 100), ("ground_train", 50),
                    ("aerial_test", 700), ("ground_test", 800)):
                path = root / (name + ".png")
                Image.new("RGB", (10, height)).save(path)
                paths[name] = path
            augmenters = build_resolution_augmenters(
                _args(), [(str(paths["aerial_train"]), 0, 0)],
                [(str(paths["ground_train"]), 0, 1)], stage=1,
                printer=lambda message: None)
            self.assertEqual(
                augmenters["aerial"].target_heights, (50,))
            self.assertEqual(
                augmenters["ground"].target_heights, (100,))

    def test_ground_resolution_process_touches_only_second_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = osp.join(directory, "sample.png")
            Image.new("RGB", (2, 2), (10, 20, 30)).save(path)

            def pixel(image):
                return image.getpixel((0, 0))

            def make_red(image, path_value):
                del path_value
                return Image.new("RGB", image.size, (255, 0, 0))

            dataset = Preprocessor_color(
                [(path, 0, 1)], transform=pixel, transform1=pixel,
                resolution_process1=make_red)
            first, second = dataset[0][:2]
            self.assertEqual(first, (10, 20, 30))
            self.assertEqual(second, (255, 0, 0))

    def test_aerial_image_process_runs_before_resolution_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = osp.join(directory, "sample.png")
            Image.new("RGB", (2, 2), (10, 20, 30)).save(path)
            observed = []

            def dagva(image, path_value):
                del path_value
                return Image.new("RGB", image.size, (100, 110, 120))

            def resolution(image, path_value):
                del path_value
                observed.append(image.getpixel((0, 0)))
                return image

            dataset = Preprocessor(
                [(path, 0, 0)], transform=lambda image: image,
                image_process=dagva, resolution_process=resolution)
            dataset[0]
            self.assertEqual(observed, [(100, 110, 120)])


class ResolutionAugmentRoutingTests(unittest.TestCase):
    def _parse(self, *items):
        return runner.build_parser().parse_args(list(items) + ["--dry-run"])

    def test_default_commands_and_log_names_have_no_resolution_changes(self):
        for dataset_key in ("agreid", "agreidv2"):
            args = self._parse(
                "--dataset-key", dataset_key, "--experiment", "baseline")
            names = runner.default_log_names(args)
            self.assertNotIn("_res", names[0])
            self.assertNotIn("_res", names[1])
            commands = (
                runner.build_stage1_command(args, *names),
                runner.build_stage2_command(
                    args, osp.join(ROOT, "stage1.pth.tar"), *names))
            for command in commands:
                self.assertFalse(any(
                    token.startswith("--res-aug-") for token in command))

    def test_enabled_commands_add_only_resolution_options_and_suffix(self):
        baseline = self._parse("--experiment", "baseline")
        enabled = self._parse(
            "--experiment", "baseline", "--res-aug-mode", "global")
        baseline_names = runner.default_log_names(baseline)
        enabled_names = runner.default_log_names(enabled)
        self.assertEqual(
            enabled_names[0], baseline_names[0] + "_resglobal_p0.5")
        self.assertEqual(
            enabled_names[1], baseline_names[1] + "_resglobal_p0.5")
        command = runner.build_stage1_command(enabled, *enabled_names)
        added = [token for token in command
                 if token not in runner.build_stage1_command(
                     baseline, *baseline_names)]
        self.assertIn("--res-aug-mode", added)
        self.assertIn("global", added)
        self.assertTrue(all(
            token.startswith("--res-aug-") or not token.startswith("--")
            for token in added))

    def test_nondefault_resolution_values_extend_suffix(self):
        args = self._parse(
            "--res-aug-mode", "multi", "--res-aug-prob", "0.7",
            "--res-aug-domains", "ground", "--res-aug-stages", "2",
            "--res-aug-min-scale", "0.2", "--res-aug-patches", "2", "4",
            "--res-aug-patch-area", "0.1", "0.3")
        stage1, stage2 = runner.default_log_names(args)
        suffix = "_resmulti_p0.7_domground_st2_min0.2_k2-4_a0.1-0.3"
        self.assertTrue(stage1.endswith(suffix))
        self.assertTrue(stage2.endswith(suffix))
        checkpoint = runner.checkpoint_path(args, stage1)
        self.assertIn(stage1, checkpoint)


class ResolutionBucketTests(unittest.TestCase):
    def test_bucket_evaluation_does_not_change_overall_metrics(self):
        distmat = np.asarray([
            [0.1, 0.9, 1.0, 0.2, 0.8, 1.1],
            [0.9, 0.1, 1.0, 0.8, 0.2, 1.1],
            [0.2, 0.8, 1.0, 0.1, 0.9, 1.1],
            [0.8, 0.2, 1.0, 0.9, 0.1, 1.1],
        ], dtype=np.float64)
        query_pids = np.asarray([1, 2, 1, 2])
        gallery_pids = np.asarray([1, 2, 3, 1, 2, 3])
        query_camids = np.zeros(4, dtype=np.int64)
        gallery_camids = np.ones(6, dtype=np.int64)
        before_matrix = distmat.copy()
        before = eval_agreid(
            distmat, query_pids, gallery_pids,
            q_camids=query_camids, g_camids=gallery_camids)
        rows = evaluate_resolution_buckets(
            distmat, query_pids, gallery_pids,
            query_camids, gallery_camids,
            np.asarray([60, 100, 160, 200]), (80, 150), eval_agreid)
        after = eval_agreid(
            distmat, query_pids, gallery_pids,
            q_camids=query_camids, g_camids=gallery_camids)
        np.testing.assert_array_equal(distmat, before_matrix)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertEqual(before[2], after[2])
        self.assertEqual([row["query_images"] for row in rows], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
