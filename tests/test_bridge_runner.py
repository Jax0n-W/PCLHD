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
        self.assertIn("--log-gt-match-accuracy", command)

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
        self.assertIn("_train_agreid_bridge_engine.py", text)
        self.assertIn(
            "agreidv2_dt_bridge_d0_b1_channel_off_s2_seed1_eps0.6",
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
