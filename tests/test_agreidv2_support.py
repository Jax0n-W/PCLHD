from __future__ import absolute_import

import os.path as osp
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiment_runner as runner
from agreid_identity import parse_agreidv2_manifest
from examples.PCLHD_AGReIDv2_test import modal_for_camid


def parse(*items):
    return runner.build_parser().parse_args(list(items) + ["--dry-run"])


def without_options(command, options):
    result = []
    index = 0
    while index < len(command):
        if command[index] in options:
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


class AGReIDV2SupportTests(unittest.TestCase):
    def test_manifest_split_and_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            inner = outer / "AG-ReID.v2"
            (inner / "query" / "0001").mkdir(parents=True)
            (inner / "gallery" / "0001").mkdir(parents=True)
            query_name = "P0001T04041A0C3F121.jpg"
            gallery_name = "P0001T04041A0C0F122.jpg"
            manifest = outer / "exp4_cctv_to_aerial.txt"
            manifest.write_text(
                "query/0001/{}\ngallery/0001/{}\n".format(
                    query_name, gallery_name), encoding="utf-8")

            query, gallery = parse_agreidv2_manifest(
                directory, "exp4", validate=False)
            self.assertEqual(len(query), 1)
            self.assertEqual(len(gallery), 1)
            self.assertEqual(
                query[0].path,
                osp.normpath(str(inner / "query" / "0001" / query_name)))
            self.assertEqual(
                gallery[0].path,
                osp.normpath(str(
                    inner / "gallery" / "0001" / gallery_name)))
            self.assertEqual(query[0].camid, 3)
            self.assertEqual(gallery[0].camid, 0)

    def test_modal_mapping(self):
        self.assertEqual(modal_for_camid(0), 2)
        self.assertEqual(modal_for_camid(2), 1)
        self.assertEqual(modal_for_camid(3), 1)
        with self.assertRaises(ValueError):
            modal_for_camid(1)

    def test_v2_baseline_commands_only_change_dataset_and_logs(self):
        v1 = parse("--dataset-key", "agreid", "--experiment", "baseline")
        v2 = parse("--dataset-key", "agreidv2", "--experiment", "baseline")
        v1_s1, v1_s2 = runner.default_log_names(v1)
        v2_s1, v2_s2 = runner.default_log_names(v2)
        ignored = {
            "--dataset", "--dataset-key", "--stage1-log-name",
            "--stage2-log-name",
        }
        self.assertEqual(
            without_options(
                runner.build_stage1_command(v1, v1_s1, v1_s2), ignored),
            without_options(
                runner.build_stage1_command(v2, v2_s1, v2_s2), ignored))
        checkpoint = osp.join(ROOT, "synthetic_stage1.pth.tar")
        self.assertEqual(
            without_options(
                runner.build_stage2_command(
                    v1, checkpoint, v1_s1, v1_s2), ignored),
            without_options(
                runner.build_stage2_command(
                    v2, checkpoint, v2_s1, v2_s2), ignored))
        self.assertTrue(v1_s1.startswith("agreid_dt_"))
        self.assertTrue(v2_s1.startswith("agreidv2_dt_"))

    def test_v2_accepts_all_experiments_in_dry_run(self):
        for experiment in (
                "baseline", "dynamic", "three_domain", "full",
                "bridge", "dynamic_bridge"):
            args = parse(
                "--dataset-key", "agreidv2", "--experiment", experiment,
                "--data-dir", "/data/AG-ReID.v2")
            runner.validate_args(args)

    def test_v2_default_paths_follow_data_dir(self):
        args = parse("--dataset-key", "agreidv2", "--experiment", "full",
                     "--data-dir", "/data/AG-ReID.v2/")
        runner.resolve_dataset_paths(args)
        outer = runner._path("/data/AG-ReID.v2")
        self.assertEqual(args.dynamic_agva_mask_dir,
                         osp.join(outer + "_SCHP", "train_all"))
        self.assertEqual(args.dynamic_agva_image_root,
                         osp.join(outer, "AG-ReID.v2", "train_all"))
        self.assertEqual(args.view_score_file, runner.DEFAULT_V2_VIEW_SCORE)

    def test_v1_default_paths_are_unchanged(self):
        args = parse("--experiment", "full")
        runner.resolve_dataset_paths(args)
        self.assertEqual(args.dynamic_agva_mask_dir, runner.DEFAULT_MASK_DIR)
        self.assertEqual(args.dynamic_agva_image_root,
                         runner.DEFAULT_IMAGE_ROOT)
        self.assertEqual(args.view_score_file, runner.DEFAULT_VIEW_SCORE)

    def test_explicit_paths_override_v2_defaults(self):
        args = parse("--dataset-key", "agreidv2", "--experiment", "full",
                     "--dynamic-agva-mask-dir", "/m",
                     "--dynamic-agva-image-root", "/i",
                     "--view-score-file", "/v.csv")
        runner.resolve_dataset_paths(args)
        self.assertEqual(args.dynamic_agva_mask_dir, "/m")
        self.assertEqual(args.dynamic_agva_image_root, "/i")
        self.assertEqual(args.view_score_file, "/v.csv")

    def test_v2_dynamic_stage1_command(self):
        args = parse("--dataset-key", "agreidv2", "--experiment", "dynamic",
                     "--data-dir", "/data/AG-ReID.v2", "--eps", "0.6")
        stage1_name, stage2_name = runner.default_log_names(args)
        self.assertEqual(stage1_name,
                         "agreidv2_dt_s1_d1_channel_off_seed1_eps0.6")
        self.assertEqual(
            stage2_name,
            "agreidv2_dt_dynamic_d1_t0_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage1_command(args)
        self.assertIn("--use-dynamic-agva", command)
        outer = runner._path("/data/AG-ReID.v2")
        self.assertEqual(
            command[command.index("--dynamic-agva-image-root") + 1],
            osp.join(outer, "AG-ReID.v2", "train_all"))
        self.assertEqual(command[command.index("--dataset") + 1],
                         "agreidv2_ir")
        self.assertEqual(command[command.index("--dataset-key") + 1],
                         "agreidv2")

    def test_v2_three_domain_stage2_command(self):
        args = parse("--dataset-key", "agreidv2",
                     "--experiment", "three_domain",
                     "--data-dir", "/data/AG-ReID.v2", "--eps", "0.6",
                     "--dbscan-backend", "fast")
        _, stage2_name = runner.default_log_names(args)
        self.assertEqual(
            stage2_name,
            "agreidv2_dt_three_domain_d0_t1_channel_off_s2_seed1_eps0.6")
        command = runner.build_stage2_command(args, "/tmp/s1.pth.tar")
        self.assertTrue(
            command[1].endswith("_train_agreid_three_domain_engine.py"))
        self.assertEqual(command[command.index("--dataset") + 1],
                         "agreidv2_ir")
        self.assertEqual(command[command.index("--dataset-key") + 1],
                         "agreidv2")
        self.assertEqual(command[command.index("--view-score-file") + 1],
                         runner.DEFAULT_V2_VIEW_SCORE)
        self.assertEqual(command[command.index("--dbscan-backend") + 1],
                         "fast")
        self.assertNotIn("--use-dynamic-agva", command)

    def test_v1_three_domain_command_has_no_dataset_key(self):
        args = parse("--experiment", "three_domain")
        command = runner.build_stage2_command(args, "/tmp/s1.pth.tar")
        self.assertEqual(command[command.index("--dataset") + 1],
                         "agreid_ir")
        self.assertNotIn("--dataset-key", command)


if __name__ == "__main__":
    unittest.main()
