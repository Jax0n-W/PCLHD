from __future__ import absolute_import

import contextlib
import io
import os.path as osp
import sys
import unittest


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner


def parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


class BridgeRunnerTests(unittest.TestCase):

    def test_bridge_log_names_and_command(self):
        args = parse(
            "--dataset-key", "agreidv2", "--experiment", "bridge",
            "--eps", "0.6", "--dbscan-backend", "fast",
            "--bridge-score", "random", "--bridge-topk", "7",
            "--bridge-ratio", "0.4", "--log-gt-match-accuracy")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage1, "agreidv2_dt_s1_d0_channel_off_seed1_eps0.6")
        self.assertEqual(
            stage2,
            "agreidv2_dt_bridge_d0_b1_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage2_command(
            args, "/tmp/model_best.pth.tar", stage1, stage2)
        self.assertTrue(command[1].endswith(
            "_train_agreid_bridge_engine.py"))
        self.assertEqual(command[command.index("--bridge-score") + 1],
                         "random")
        self.assertEqual(command[command.index("--bridge-topk") + 1], "7")
        self.assertEqual(command[command.index("--bridge-ratio") + 1], "0.4")
        self.assertEqual(command[command.index("--bgm-fallback") + 1],
                         "pclhd")
        self.assertIn("--log-gt-match-accuracy", command)

    def test_nearest_fallback_adds_log_suffix_and_is_forwarded(self):
        args = parse(
            "--dataset-key", "agreidv2", "--experiment", "bridge",
            "--eps", "0.6", "--bgm-fallback", "nearest")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage2,
            "agreidv2_dt_bridge_d0_b1_nnfb_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage2_command(
            args, "/tmp/model_best.pth.tar", stage1, stage2)
        self.assertEqual(command[command.index("--bgm-fallback") + 1],
                         "nearest")

    def test_dynamic_bridge_reuses_dynamic_stage1(self):
        args = parse(
            "--dataset-key", "agreidv2",
            "--experiment", "dynamic_bridge", "--eps", "0.6")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage1, "agreidv2_dt_s1_d1_channel_off_seed1_eps0.6")
        self.assertEqual(
            stage2,
            "agreidv2_dt_dynamic_bridge_d1_b1_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage1_command(args, stage1, stage2)
        self.assertIn("--use-dynamic-agva", command)

    def test_confident_names_fixed_bridge_settings_and_parameters(self):
        args = parse(
            "--dataset-key", "agreidv2", "--experiment", "confident",
            "--eps", "0.6", "--cross-retain-start", "0.3",
            "--cross-retain-end", "0.9",
            "--cross-retain-ramp-epochs", "12",
            "--cross-confidence", "similarity",
            "--bridge-ratio", "0.25")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage1, "agreidv2_dt_s1_d0_channel_off_seed1_eps0.6")
        self.assertEqual(
            stage2,
            "agreidv2_dt_confident_d0_cr0.3-0.9r12_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage2_command(
            args, "/tmp/model_best.pth.tar", stage1, stage2)
        self.assertIn("--confident-cross", command)
        self.assertEqual(command[command.index("--bridge-ratio") + 1], "1.0")
        self.assertEqual(
            command[command.index("--bgm-fallback") + 1], "pclhd")
        self.assertEqual(
            command[command.index("--cross-retain-start") + 1], "0.3")
        self.assertEqual(
            command[command.index("--cross-retain-end") + 1], "0.9")
        self.assertEqual(
            command[command.index("--cross-retain-ramp-epochs") + 1],
            "12")
        self.assertEqual(
            command[command.index("--cross-confidence") + 1],
            "similarity")

    def test_confident_default_fallback_and_original_log_names(self):
        names_by_dataset = {
            "agreid": (
                "agreid_dt_s1_d0_channel_off_seed1_eps0.6",
                "agreid_dt_confident_d0_cr0.4-1.0r20_channel_off_s2_seed1_eps0.6"),
            "agreidv2": (
                "agreidv2_dt_s1_d0_channel_off_seed1_eps0.6",
                "agreidv2_dt_confident_d0_cr0.4-1.0r20_channel_off_s2_seed1_eps0.6"),
        }
        for dataset, names in names_by_dataset.items():
            with self.subTest(dataset=dataset):
                args = parse("--dataset-key", dataset,
                             "--experiment", "confident", "--eps", "0.6")
                self.assertEqual(runner.default_log_names(args), names)
                command = runner.build_stage2_command(
                    args, "/tmp/model_best.pth.tar")
                self.assertEqual(
                    command[command.index("--bgm-fallback") + 1], "pclhd")

    def test_confident_experiments_forward_nearest_with_fixed_ratio(self):
        names_by_experiment = {
            "confident":
                "agreid_dt_confident_d0_cr0.4-1.0r20_nnfb_channel_off_s2_seed1_eps0.6",
            "dynamic_confident":
                "agreid_dt_dynamic_confident_d1_cr0.4-1.0r20_nnfb_channel_off_s2_seed1_eps0.6",
        }
        for experiment, expected in names_by_experiment.items():
            with self.subTest(experiment=experiment):
                args = parse("--experiment", experiment, "--eps", "0.6",
                             "--bgm-fallback", "nearest", "--bridge-ratio",
                             "0.25", "--log-gt-match-accuracy")
                self.assertEqual(runner.default_log_names(args)[1], expected)
                command = runner.build_stage2_command(
                    args, "/tmp/model_best.pth.tar")
                self.assertEqual(
                    command[command.index("--bgm-fallback") + 1], "nearest")
                self.assertEqual(
                    command[command.index("--bridge-ratio") + 1], "1.0")
                self.assertIn("--log-gt-match-accuracy", command)

    def test_dynamic_confident_reuses_dynamic_stage1(self):
        args = parse(
            "--dataset-key", "agreidv2",
            "--experiment", "dynamic_confident", "--eps", "0.6")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage1, "agreidv2_dt_s1_d1_channel_off_seed1_eps0.6")
        self.assertEqual(
            stage2,
            "agreidv2_dt_dynamic_confident_d1_cr0.4-1.0r20_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage1_command(args, stage1, stage2)
        self.assertIn("--use-dynamic-agva", command)

    def test_v1_bridge_uses_v1_dataset_without_dataset_key_override(self):
        args = parse("--experiment", "bridge", "--eps", "0.6")
        stage1, stage2 = runner.default_log_names(args)
        self.assertEqual(
            stage2, "agreid_dt_bridge_d0_b1_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage2_command(
            args, "/tmp/model_best.pth.tar", stage1, stage2)
        self.assertEqual(command[command.index("--dataset") + 1],
                         "agreid_ir")
        self.assertNotIn("--dataset-key", command)

    def test_dry_run_prints_bridge_config_and_routes_engine(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.run_experiment("agreidv2", argv=[
                "--experiment", "bridge", "--run-mode", "stage2",
                "--stage1-checkpoint", "/tmp/model_best.pth.tar",
                "--bridge-score", "ground_knn", "--eps", "0.6",
                "--dry-run"])
        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Bridge matching (Stage 2): True", text)
        self.assertIn("Bridge score: ground_knn", text)
        self.assertIn("BGM fallback: pclhd", text)
        self.assertIn("_train_agreid_bridge_engine.py", text)
        self.assertIn(
            "agreidv2_dt_bridge_d0_b1_channel_off_s2_seed1_eps0.6",
            text)

    def test_confident_dry_run_prints_configuration(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.run_experiment("agreidv2", argv=[
                "--experiment", "confident", "--run-mode", "stage2",
                "--stage1-checkpoint", "/tmp/model_best.pth.tar",
                "--eps", "0.6", "--dry-run"])
        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Confidence-aware cross matching (Stage 2): True", text)
        self.assertIn("Fixed bridge ratio: 1.0", text)
        self.assertIn("BGM fallback: pclhd", text)
        self.assertIn("Cross confidence: margin", text)
        self.assertIn(
            "agreidv2_dt_confident_d0_cr0.4-1.0r20_channel_off_s2_seed1_eps0.6",
            text)

    def test_legacy_log_names_and_commands_have_no_bridge_tokens(self):
        expected = {
            "baseline": (
                "agreid_dt_s1_d0_channel_off_seed1",
                "agreid_dt_baseline_d0_t0_channel_off_s2_seed1"),
            "dynamic": (
                "agreid_dt_s1_d1_channel_off_seed1",
                "agreid_dt_dynamic_d1_t0_channel_off_s2_seed1"),
            "three_domain": (
                "agreid_dt_s1_d0_channel_off_seed1",
                "agreid_dt_three_domain_d0_t1_channel_off_s2_seed1"),
            "full": (
                "agreid_dt_s1_d1_channel_off_seed1",
                "agreid_dt_full_d1_t1_channel_off_s2_seed1"),
        }
        for experiment, names in expected.items():
            args = parse("--experiment", experiment)
            self.assertEqual(runner.default_log_names(args), names)
            stage1 = runner.build_stage1_command(args, *names)
            stage2 = runner.build_stage2_command(
                args, "/tmp/model_best.pth.tar", *names)
            self.assertNotIn("--bridge-score", stage1)
            self.assertNotIn("--bridge-score", stage2)
            self.assertNotIn("--cluster-lock-file", stage1)
            self.assertNotIn("--cluster-lock-file", stage2)


if __name__ == "__main__":
    unittest.main()
