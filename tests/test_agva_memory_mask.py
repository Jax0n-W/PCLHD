from __future__ import absolute_import

import argparse
import hashlib
import json
import os.path as osp
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner  # noqa: E402
from clustercontrast.models.cm import ClusterMemory  # noqa: E402
from clustercontrast.models.cm_masked import MaskedClusterMemory  # noqa: E402
from clustercontrast.utils.data.agva_flag import (  # noqa: E402
    DynamicAGVAProcessWithFlag, adaptive_strength_scale,
    mask_geometry_ratio)
from clustercontrast.utils.data.dynamic_agva import (  # noqa: E402
    DynamicAGVAProcess)


def _parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


class _CPUCUDA(object):
    def __enter__(self):
        self.tensor = mock.patch.object(
            torch.Tensor, "cuda", lambda value, *args, **kwargs: value)
        self.module = mock.patch.object(
            torch.nn.Module, "cuda", lambda value, *args, **kwargs: value)
        self.tensor.start()
        self.module.start()

    def __exit__(self, *unused):
        self.module.stop()
        self.tensor.stop()


def _memory(mode):
    memory = ClusterMemory(
        4, 3, temp=0.2, momentum=0.2, mode=mode, num_instances=2)
    masked = MaskedClusterMemory(
        4, 3, temp=0.2, momentum=0.2, mode=mode, num_instances=2)
    torch.manual_seed(4)
    with torch.no_grad():
        memory.features.copy_(torch.nn.functional.normalize(
            torch.randn_like(memory.features), dim=1))
        if mode == "CMhard":
            memory.features_ema.copy_(torch.nn.functional.normalize(
                torch.randn_like(memory.features_ema), dim=1))
    masked.load_state_dict(memory.state_dict())
    return memory, masked


class MaskedMemoryTests(unittest.TestCase):
    def _compare(self, mode, model_name, update_mask):
        with _CPUCUDA():
            original, masked = _memory(mode)
            torch.manual_seed(7)
            x1 = torch.randn(6, 4, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_(True)
            targets = torch.tensor([0, 1, 2, 0, 1, 2])
            np.random.seed(19)
            loss1 = original(x1, targets, model_name=model_name)
            loss1.backward()
            np_state1 = np.random.get_state()
            np.random.seed(19)
            loss2 = masked(
                x2, targets, model_name=model_name,
                update_mask=update_mask)
            loss2.backward()
            np_state2 = np.random.get_state()
            self.assertTrue(torch.equal(loss1, loss2))
            self.assertTrue(torch.equal(x1.grad, x2.grad))
            self.assertTrue(torch.equal(original.features, masked.features))
            if mode == "CMhard":
                self.assertTrue(torch.equal(
                    original.features_ema, masked.features_ema))
            self.assertEqual(np_state1[0], np_state2[0])
            self.assertTrue(np.array_equal(np_state1[1], np_state2[1]))
            self.assertEqual(np_state1[2:], np_state2[2:])

    def test_none_is_bit_exact_for_all_memory_types(self):
        self._compare("CM", "encoder", None)
        self._compare("CMhybrid", "encoder", None)
        self._compare("CMhard", "encoder", None)
        self._compare("CMhard", "encoder_ema", None)

    def test_all_true_is_bit_exact_for_all_memory_types(self):
        mask = torch.ones(6, dtype=torch.bool)
        self._compare("CM", "encoder", mask)
        self._compare("CMhybrid", "encoder", mask)
        self._compare("CMhard", "encoder", mask)
        self._compare("CMhard", "encoder_ema", mask)

    def test_excluded_samples_keep_class_slots_but_keep_input_gradient(self):
        with _CPUCUDA():
            for mode, model_name in (
                    ("CM", "encoder"), ("CMhybrid", "encoder"),
                    ("CMhard", "encoder"),
                    ("CMhard", "encoder_ema")):
                _, memory = _memory(mode)
                before_features = memory.features.clone()
                before_ema = (memory.features_ema.clone()
                              if mode == "CMhard" else None)
                inputs = torch.tensor([
                    [1.0, 0.2, 0.1, 0.0],
                    [0.1, 1.0, 0.2, 0.0],
                    [0.0, 0.2, 1.0, 0.1],
                ], requires_grad=True)
                targets = torch.tensor([0, 1, 1])
                update = torch.tensor([False, True, False])
                np.random.seed(3)
                memory(inputs, targets, model_name=model_name,
                       update_mask=update).backward()
                self.assertGreater(float(inputs.grad[0].abs().sum()), 0.0)
                if model_name == "encoder_ema":
                    # Every instance slot belonging to excluded class 0 stays.
                    self.assertTrue(torch.equal(
                        memory.features_ema[[0, 3]],
                        before_ema[[0, 3]]))
                elif mode == "CM":
                    self.assertTrue(torch.equal(
                        memory.features[0], before_features[0]))
                else:
                    self.assertTrue(torch.equal(
                        memory.features[[0, 3]], before_features[[0, 3]]))


class AGVAFlagTests(unittest.TestCase):
    @staticmethod
    def _fixture(directory):
        root = Path(directory) / "images"
        masks = Path(directory) / "masks"
        root.mkdir()
        masks.mkdir()
        path = root / "person.jpg"
        image = Image.fromarray(np.tile(
            np.arange(64, dtype=np.uint8)[:, None, None], (1, 40, 3)))
        image.save(path)
        mask = np.zeros((64, 40), dtype=np.uint8)
        mask[8:58, 6:34] = 255
        Image.fromarray(mask).save(masks / "person.png")
        return image, path, masks, mask

    def test_fixed_mode_matches_original_random_stream_and_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            image, path, masks, _ = self._fixture(directory)
            original = DynamicAGVAProcess(
                str(masks), image_root=str(path.parent), apply_slots=2,
                debug=True)
            flagged = DynamicAGVAProcessWithFlag(
                str(masks), image_root=str(path.parent), apply_slots=2,
                target="fixed")
            random.seed(31)
            expected, info = original(image, str(path))
            expected_state = random.getstate()
            random.seed(31)
            actual, applied = flagged(image, str(path))
            actual_state = random.getstate()
            self.assertTrue(np.array_equal(np.asarray(expected),
                                           np.asarray(actual)))
            self.assertEqual(applied, info.applied)
            self.assertEqual(expected_state, actual_state)

    def test_probability_skip_and_missing_mask_are_false(self):
        with tempfile.TemporaryDirectory() as directory:
            image, path, masks, _ = self._fixture(directory)
            process = DynamicAGVAProcessWithFlag(
                str(masks), image_root=str(path.parent), apply_slots=1)
            with mock.patch("random.randint", return_value=3):
                _, applied = process(image, str(path))
            self.assertFalse(applied)
            missing = path.parent / "missing.jpg"
            with mock.patch("random.randint", return_value=1):
                _, applied = process(image, str(missing))
            self.assertFalse(applied)

    def test_ground_mixture_equal_geometry_never_augments(self):
        with tempfile.TemporaryDirectory() as directory:
            image, path, masks, mask = self._fixture(directory)
            own = mask_geometry_ratio(Image.fromarray(mask), image.size)
            process = DynamicAGVAProcessWithFlag(
                str(masks), image_root=str(path.parent), apply_slots=1,
                target="ground-mixture", need_scale=0.2,
                ground_ratios=[own, own, own])
            for seed in range(20):
                random.seed(seed)
                with mock.patch("random.randint", return_value=1):
                    actual, applied = process(image, str(path))
                self.assertFalse(applied)
                self.assertTrue(np.array_equal(
                    np.asarray(image), np.asarray(actual)))

    def test_adaptive_scale_zero_and_monotonic(self):
        own = 2.0
        self.assertEqual(adaptive_strength_scale(own, 1.9, 0.2), 0.0)
        self.assertEqual(adaptive_strength_scale(own, own, 0.2), 0.0)
        values = [adaptive_strength_scale(own, ref, 0.2)
                  for ref in (2.02, 2.1, 2.2, 2.4, 3.0)]
        self.assertEqual(values, sorted(values))


class AGVARunnerTests(unittest.TestCase):
    @staticmethod
    def _command_snapshot(command):
        path_flags = {
            "--data-dir", "--logs-dir", "--dynamic-agva-mask-dir",
            "--dynamic-agva-image-root", "--view-score-file",
            "--stage1-checkpoint"}
        normalized = []
        for index, value in enumerate(command):
            if index == 0:
                value = "<PYTHON>"
            elif index == 1:
                value = osp.basename(value)
            elif command[index - 1] in path_flags:
                value = "<PATH>"
            else:
                value = value.replace("\\", "/")
            normalized.append(value)
        payload = json.dumps(
            normalized, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def test_existing_experiment_names_and_commands_have_no_new_tokens(self):
        expected_names = {
            "baseline": ("agreid_dt_s1_d0_channel_off_seed1",
                         "agreid_dt_baseline_d0_t0_channel_off_s2_seed1"),
            "dynamic": ("agreid_dt_s1_d1_channel_off_seed1",
                        "agreid_dt_dynamic_d1_t0_channel_off_s2_seed1"),
            "three_domain": ("agreid_dt_s1_d0_channel_off_seed1",
                             "agreid_dt_three_domain_d0_t1_channel_off_s2_seed1"),
            "full": ("agreid_dt_s1_d1_channel_off_seed1",
                     "agreid_dt_full_d1_t1_channel_off_s2_seed1"),
            "bridge": ("agreid_dt_s1_d0_channel_off_seed1",
                       "agreid_dt_bridge_d0_b1_channel_off_s2_seed1"),
            "dynamic_bridge": ("agreid_dt_s1_d1_channel_off_seed1",
                               "agreid_dt_dynamic_bridge_d1_b1_channel_off_s2_seed1"),
            "confident": ("agreid_dt_s1_d0_channel_off_seed1",
                          "agreid_dt_confident_d0_cr0.4-1.0r20_channel_off_s2_seed1"),
            "dynamic_confident": (
                "agreid_dt_s1_d1_channel_off_seed1",
                "agreid_dt_dynamic_confident_d1_cr0.4-1.0r20_channel_off_s2_seed1"),
        }
        forbidden = {
            "--dynamic-agva-no-memory-update", "--dynamic-agva-target",
            "--dynamic-agva-need-scale", "--dynamic-agva-ground-mask-dir"}
        for experiment, names in expected_names.items():
            args = _parse("--experiment", experiment)
            self.assertEqual(runner.default_log_names(args), names)
            stage1 = runner.build_stage1_command(args, *names)
            stage2 = runner.build_stage2_command(
                args, "/tmp/model_best.pth.tar", *names)
            self.assertFalse(forbidden.intersection(stage1))
            self.assertFalse(forbidden.intersection(stage2))

    def test_all_d5a4f86_command_snapshots_are_exact(self):
        # Generated once from d5a4f86 with path-bearing values normalized.
        snapshots = {
            "baseline": (
                "e86d1d364c691cb63167aecceb9fcf4636c35e68e6bb708843979f712b5724c6",
                "fab294dc24bbccd0819008eeffaa0060b5e1ac93fb57239fd7f3c46ffbfeb6ee"),
            "dynamic": (
                "200cafdb1d8c922eaf0a902e877e279293d2a47e00fbce9ac9364cd3398f3099",
                "ea6cb841f5e31194292666d47f85ac8c87dcf1e49a43d6d4f6c6cec9504b9efa"),
            "three_domain": (
                "a0f95fa151c1c1abb7a28ff1fb4f1caeccd654827e73dd9647d2ce7091f4479c",
                "bf4719f68e835723a447d06d051510057b74ac0f76863588f6bbb4b898f1ad23"),
            "full": (
                "40e80d31c19f73e445ba31c89514b90631506b2e43a0581a05e38e937e0dd23f",
                "67f4fd43c9d3cb4eb57c9448ad220d664d7fa49cafe6a1e6edf79352caeb21bb"),
            "bridge": (
                "e418ddd66faff783315f38b67b6267bf118718f70bdfd5ed29e585d45774e130",
                "e7767c04fedb30a8f2c07ee150d475488708073b4bb9ecd59e1b08859b32811c"),
            "dynamic_bridge": (
                "5d011b6e06a373936c4d33bb5fa0ba9e54712e3dd302f64e748efba8470fb1e6",
                "c14029001f418102d10c8880b32ee2af05984b482908f08cffccd637258d6f4d"),
            "confident": (
                "b7a17723eb7dbcfd5fc15ab2e65a07c835d70a2821a16fc438ab66cc10041ab6",
                "c6a05a4bf887139a7443516423666d2aaa2b4f940ea1802988aaea1ee933b48b"),
            "dynamic_confident": (
                "07dfbcb568605d69e5c23fa75c49ad9cd710a541bb117cecd024eefbb1f6013c",
                "38ae31732c4498bb58079b8e9ed3e4b5b6a9091b6f46de8c03ee899e359af1bb"),
        }
        for experiment, expected in snapshots.items():
            args = _parse("--experiment", experiment)
            stage1 = runner.build_stage1_command(args)
            stage2 = runner.build_stage2_command(
                args, "/tmp/model_best.pth.tar")
            self.assertEqual(self._command_snapshot(stage1), expected[0])
            self.assertEqual(self._command_snapshot(stage2), expected[1])

    def test_new_tags_and_arguments_are_isolated(self):
        args = _parse(
            "--dataset-key", "agreidv2", "--experiment", "dynamic",
            "--eps", "0.6", "--dynamic-agva-no-memory-update",
            "--dynamic-agva-target", "ground-mixture")
        runner.validate_args(args)
        names = runner.default_log_names(args)
        self.assertEqual(
            names[0],
            "agreidv2_dt_s1_d1mmgm0.2_channel_off_seed1_eps0.6")
        self.assertEqual(
            names[1],
            "agreidv2_dt_dynamic_d1mmgm0.2_t0_channel_off_s2_seed1_eps0.6")
        stage1 = runner.build_stage1_command(args, *names)
        self.assertIn("--use-dynamic-agva", stage1)
        self.assertIn("--dynamic-agva-no-memory-update", stage1)
        self.assertIn("--dynamic-agva-target", stage1)
        self.assertIn("--dynamic-agva-ground-mask-dir", stage1)

        stage2 = runner.build_stage2_command(
            args, "/tmp/model_best.pth.tar", *names)
        forbidden = {
            "--use-dynamic-agva", "--dynamic-agva-no-memory-update",
            "--dynamic-agva-target", "--dynamic-agva-need-scale",
            "--dynamic-agva-ground-mask-dir"}
        self.assertFalse(forbidden.intersection(stage2))

        baseline_args = _parse(
            "--dataset-key", "agreidv2", "--experiment", "dynamic",
            "--eps", "0.6")
        baseline_stage2 = runner.build_stage2_command(
            baseline_args, "/tmp/model_best.pth.tar")

        def without_log_names(command):
            command = list(command)
            for option in ("--stage1-log-name", "--stage2-log-name"):
                index = command.index(option)
                del command[index:index + 2]
            return command

        self.assertEqual(
            without_log_names(stage2),
            without_log_names(baseline_stage2))

    def test_non_dynamic_and_other_trainers_reject_new_switches(self):
        with self.assertRaisesRegex(ValueError, "dynamic experiment"):
            runner.validate_args(_parse(
                "--experiment", "baseline",
                "--dynamic-agva-target", "fixed"))
        for experiment in ("full", "dynamic_bridge", "dynamic_confident"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                runner.validate_args(_parse(
                    "--experiment", experiment,
                    "--dynamic-agva-no-memory-update"))


if __name__ == "__main__":
    unittest.main()
