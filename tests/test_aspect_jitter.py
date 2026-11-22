from __future__ import absolute_import

import os.path as osp
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner  # noqa: E402
from clustercontrast.utils.data.aspect_geometry import (  # noqa: E402
    GeometryAspectProcess, _PreprocessorColorGeometry,
    _PreprocessorGeometry, compute_geometry_gaps)
from clustercontrast.utils.data.aspect_jitter import (  # noqa: E402
    AspectJitter, apply_horizontal_ratio, build_aspect_train_transform,
    effective_aspect_max_ratio)
from clustercontrast.utils.data import transforms as T  # noqa: E402


def _parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


def _band_image(width=51, height=31, band_width=7):
    array = np.full((height, width, 3), 64, dtype=np.uint8)
    left = (width - band_width) // 2
    array[:, left:left + band_width] = (240, 20, 20)
    return Image.fromarray(array)


def _red_width(image):
    array = np.asarray(image)
    red = (array[..., 0] > 150) & (array[..., 1] < 80)
    return int(np.count_nonzero(red.any(axis=0)))


class AspectJitterTransformTests(unittest.TestCase):
    def test_probability_zero_is_pixel_exact_and_size_is_preserved(self):
        image = _band_image()
        expected = np.asarray(image)
        process = AspectJitter(max_ratio=0.2, prob=0.0, direction="both")
        for seed in range(20):
            random.seed(seed)
            actual = process(image)
            self.assertEqual(actual.size, image.size)
            self.assertTrue(np.array_equal(np.asarray(actual), expected))
        for ratio in (0.5, 0.9, 1.0, 1.1, 1.8):
            self.assertEqual(apply_horizontal_ratio(image, ratio).size,
                             image.size)

    def test_center_band_proves_narrow_and_widen_directions(self):
        image = _band_image()
        original_width = _red_width(image)
        narrow = apply_horizontal_ratio(image, 0.7)
        widen = apply_horizontal_ratio(image, 1.4)
        self.assertLess(_red_width(narrow), original_width)
        self.assertGreater(_red_width(widen), original_width)
        narrow_array = np.asarray(narrow)
        widen_array = np.asarray(widen)
        self.assertTrue(np.all(narrow_array[:, 0] == 0))
        self.assertTrue(np.all(narrow_array[:, -1] == 0))
        self.assertTrue(np.all(widen_array[:, 0] != 0))
        self.assertTrue(np.all(widen_array[:, -1] != 0))

    def test_log_uniform_samples_stay_in_directional_ranges(self):
        maximum = 0.2
        lower = 1.0 / (1.0 + maximum)
        random.seed(17)
        for direction in ("narrow", "widen", "both"):
            process = AspectJitter(maximum, 1.0, direction)
            values = [process.sample_ratio() for _ in range(2000)]
            self.assertTrue(all(lower <= value <= 1.0 + maximum
                                for value in values))
            if direction == "narrow":
                self.assertTrue(all(value <= 1.0 for value in values))
            elif direction == "widen":
                self.assertTrue(all(value >= 1.0 for value in values))
            else:
                self.assertTrue(any(value < 1.0 for value in values))
                self.assertTrue(any(value > 1.0 for value in values))

    def test_fixed_jitter_is_between_resize_and_pad(self):
        args = _parse("--aspect-jitter", "view")
        transform = build_aspect_train_transform(
            args, 1, "aerial", 288, 144,
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]))
        names = [type(operation).__name__
                 for operation in transform.transforms]
        self.assertEqual(names[:3], ["Resize", "AspectJitter", "Pad"])


class AspectGeometryTests(unittest.TestCase):
    @staticmethod
    def _save_case(root, name, box_height, box_width, with_mask=True):
        image_root = root / "images"
        mask_root = root / "masks"
        image_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        image = np.full((200, 100, 3), 120, dtype=np.uint8)
        Image.fromarray(image).save(image_root / name)
        if with_mask:
            mask = np.zeros((200, 100), dtype=np.uint8)
            top = (200 - box_height) // 2
            left = (100 - box_width) // 2
            mask[top:top + box_height, left:left + box_width] = 255
            Image.fromarray(mask).save(mask_root / Path(name).with_suffix(
                ".png"))
        return (name, 0, 0), image_root, mask_root

    def test_synthetic_masks_produce_zero_and_fifteen_percent_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aerial, aerial_root, aerial_masks = self._save_case(
                root / "aerial", "P0001T0001A0C0F001.jpg", 100, 25)
            equal, ground_root, ground_masks = self._save_case(
                root / "ground", "P0002T0001A0C2F001.jpg", 100, 25)
            slender, _, _ = self._save_case(
                root / "ground", "P0003T0001A0C3F001.jpg", 115, 25)
            result = compute_geometry_gaps(
                [aerial], [equal, slender], str(aerial_root),
                str(ground_root), str(aerial_masks), str(ground_masks),
                max_ratio=0.3, printer=lambda unused: None)
            self.assertAlmostEqual(result["gaps"][2], 0.0, places=6)
            self.assertAlmostEqual(result["gaps"][3], 0.15, places=6)

    def test_no_gap_camera_never_jitters_in_either_domain(self):
        image = _band_image()
        ground_path = "P0002T0001A0C2F001.jpg"
        ground = GeometryAspectProcess(
            "ground", {2: 0.01}, {2: 10}, prob=1.0)
        aerial = GeometryAspectProcess(
            "aerial", {2: 0.01}, {2: 10}, prob=1.0)
        expected = np.asarray(image)
        for seed in range(50):
            random.seed(seed)
            self.assertTrue(np.array_equal(
                np.asarray(ground(image, ground_path)), expected))
            self.assertTrue(np.array_equal(
                np.asarray(aerial(image, "unused.jpg")), expected))
            self.assertEqual(ground.last_ratio, 1.0)
            self.assertEqual(aerial.last_ratio, 1.0)

    def test_geometry_direction_ranges_and_output_size(self):
        image = _band_image()
        gap = 0.15
        ground = GeometryAspectProcess(
            "ground", {3: gap}, {3: 10}, prob=1.0)
        aerial = GeometryAspectProcess(
            "aerial", {3: gap}, {3: 10}, prob=1.0)
        for seed in range(100):
            random.seed(seed)
            ground_image = ground(image, "P0002T0001A0C3F001.jpg")
            aerial_image = aerial(image, "unused.jpg")
            self.assertEqual(ground_image.size, image.size)
            self.assertEqual(aerial_image.size, image.size)
            self.assertLessEqual(1.0, ground.last_ratio)
            self.assertLessEqual(ground.last_ratio, 1.0 + gap)
            self.assertLessEqual(1.0 / (1.0 + gap), aerial.last_ratio)
            self.assertLessEqual(aerial.last_ratio, 1.0)

    def test_aerial_camera_sampling_follows_training_counts(self):
        process = GeometryAspectProcess(
            "aerial", {2: 0.0, 3: 0.15}, {2: 700, 3: 300}, prob=1.0)
        random.seed(29)
        samples = [process.sample_camera() for _ in range(30000)]
        c3_fraction = samples.count(3) / float(len(samples))
        self.assertLess(abs(c3_fraction - 0.30), 0.02)

    def test_below_eighty_percent_mask_coverage_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aerial, aerial_root, aerial_masks = self._save_case(
                root / "aerial", "P0001T0001A0C0F001.jpg", 100, 25)
            ground = []
            ground_root = ground_masks = None
            for index in range(5):
                sample, ground_root, ground_masks = self._save_case(
                    root / "ground",
                    "P{:04d}T0001A0C3F001.jpg".format(index + 2),
                    115, 25, with_mask=index < 3)
                ground.append(sample)
            with self.assertRaisesRegex(RuntimeError, "coverage below"):
                compute_geometry_gaps(
                    [aerial], ground, str(aerial_root), str(ground_root),
                    str(aerial_masks), str(ground_masks), max_ratio=0.3,
                    printer=lambda unused: None)

    def test_path_aware_preprocessors_order_and_independent_ground_views(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "P0001T0001A0C0F001.jpg"
            _band_image().save(path)
            calls = []

            def agva(image, image_path):
                calls.append(("agva", osp.basename(image_path)))
                return image

            def geometry(image, image_path):
                calls.append(("geometry", osp.basename(image_path)))
                return image

            single = _PreprocessorGeometry(
                [(path.name, 0, 0)], directory, lambda image: image,
                geometry, image_process=agva)
            single[0]
            self.assertEqual([item[0] for item in calls],
                             ["agva", "geometry"])

            calls[:] = []
            ground = _PreprocessorColorGeometry(
                [(path.name, 0, 0)], directory, lambda image: image,
                lambda image: image, geometry)
            ground[0]
            self.assertEqual([item[0] for item in calls],
                             ["geometry", "geometry"])


class AspectJitterRunnerTests(unittest.TestCase):
    def test_off_adds_no_arguments_or_log_tags_to_any_experiment(self):
        for experiment in runner.EXPERIMENTS:
            args = _parse("--experiment", experiment)
            names = runner.default_log_names(args)
            commands = (
                runner.build_stage1_command(args, *names),
                runner.build_stage2_command(
                    args, "/tmp/model_best.pth.tar", *names),
            )
            self.assertNotIn("aj", names[0])
            self.assertNotIn("aj", names[1])
            for command in commands:
                self.assertFalse(any(
                    token.startswith("--aspect-jitter")
                    for token in command))

    def test_stage_routing_and_fixed_tags(self):
        stage1 = _parse(
            "--experiment", "baseline", "--aspect-jitter", "both")
        runner.validate_args(stage1)
        names = runner.default_log_names(stage1)
        self.assertIn("d0ajboth0.2p0.5_", names[0])
        self.assertIn("d0ajboth0.2p0.5_", names[1])
        self.assertIn("--aspect-jitter",
                      runner.build_stage1_command(stage1, *names))
        self.assertNotIn(
            "--aspect-jitter", runner.build_stage2_command(
                stage1, "/tmp/model_best.pth.tar", *names))

        both_stages = _parse(
            "--experiment", "baseline", "--aspect-jitter", "view",
            "--aspect-jitter-stages", "12")
        runner.validate_args(both_stages)
        names = runner.default_log_names(both_stages)
        self.assertIn("ajview0.2p0.5s12", names[0])
        self.assertIn("--aspect-jitter",
                      runner.build_stage1_command(both_stages, *names))
        self.assertIn(
            "--aspect-jitter", runner.build_stage2_command(
                both_stages, "/tmp/model_best.pth.tar", *names))

        stage2 = _parse(
            "--experiment", "bridge", "--aspect-jitter", "both",
            "--aspect-jitter-stages", "2")
        runner.validate_args(stage2)
        names = runner.default_log_names(stage2)
        self.assertIn("ajboth0.2p0.5s2", names[0])
        self.assertNotIn("--aspect-jitter",
                         runner.build_stage1_command(stage2, *names))
        self.assertIn(
            "--aspect-jitter", runner.build_stage2_command(
                stage2, "/tmp/model_best.pth.tar", *names))

    def test_geometry_defaults_tag_paths_and_all_experiments(self):
        for experiment in runner.EXPERIMENTS:
            args = _parse(
                "--dataset-key", "agreidv2", "--experiment", experiment,
                "--aspect-jitter", "geometry")
            runner.validate_args(args)
            self.assertEqual(effective_aspect_max_ratio(args), 0.3)
            names = runner.default_log_names(args)
            self.assertIn("ajgeop0.5", names[0])
            stage1 = runner.build_stage1_command(args, *names)
            self.assertIn("--aspect-jitter-ground-mask-dir", stage1)
            self.assertIn("--dynamic-agva-mask-dir", stage1)

    def test_v1_geometry_requires_an_explicit_ground_mask_directory(self):
        args = _parse("--aspect-jitter", "geometry")
        with self.assertRaisesRegex(ValueError, "required"):
            runner.validate_args(args)


if __name__ == "__main__":
    unittest.main()
