from __future__ import absolute_import

import argparse
import os.path as osp
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.preflight_agreidv2 import (  # noqa: E402
    inspect_masks, inspect_view_scores)
from view_domain.build_view_manifest import build_records  # noqa: E402
from view_domain.common import add_view_groups, parse_image_name  # noqa: E402

V2_AERIAL = "c00,c10,c20"
V2_GROUND_FOLDS = "c02,c03;c12,c13;c22,c23"


def touch(root, name):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def namespace(aerial_dir, ground_dir, aerial="", folds=""):
    return argparse.Namespace(
        aerial_dir=Path(aerial_dir), ground_dir=Path(ground_dir),
        aerial_cameras=aerial, ground_folds=folds)


class ParseImageNameTests(unittest.TestCase):
    def test_v2_official_name_maps_to_v1_style_camera_token(self):
        self.assertEqual(parse_image_name(Path("P0001T04041A0C0F121.jpg")),
                         ("1040410", "c00"))
        self.assertEqual(parse_image_name(Path("P0001T04041A1C0F121.jpg"))[1],
                         "c10")
        self.assertEqual(parse_image_name(Path("P0012T00007A2C3F5.jpg")),
                         ("12000072", "c23"))

    def test_v1_rewritten_and_lagper_names_are_unchanged(self):
        self.assertEqual(parse_image_name(Path("P0001_T04041_C10_F121.jpg")),
                         ("1040411", "c10"))
        self.assertEqual(parse_image_name(Path("0001_c3_000002.jpg")),
                         ("0001", "c3"))


class MixedRootManifestTests(unittest.TestCase):
    def _build_v2_tree(self, root):
        names = []
        for angle in range(3):
            for person in (1, 2):
                for camera in (0, 2, 3):
                    names.append("P{:04d}T0000{}A{}C{}F1.jpg".format(
                        person, angle, angle, camera))
        nested = ["sub/" + names[0]]
        for name in names[1:] + nested:
            touch(root, name)

    def test_v2_train_all_builds_three_altitude_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            self._build_v2_tree(directory)
            records = build_records(namespace(
                directory, directory, V2_AERIAL, V2_GROUND_FOLDS))
        self.assertEqual(len(records), 18)
        by_fold = {}
        for record in records:
            by_fold.setdefault(record.fold, []).append(record)
        self.assertEqual(sorted(by_fold), [0, 1, 2])
        for fold, rows in by_fold.items():
            aerial = {row.camera for row in rows if row.domain == "aerial"}
            ground = {row.camera for row in rows if row.domain == "ground"}
            self.assertEqual(aerial, {"c{}0".format(fold)})
            self.assertEqual(ground, {"c{}2".format(fold), "c{}3".format(fold)})
        pids = [{row.pid for row in by_fold[fold]} for fold in range(3)]
        self.assertFalse(pids[0] & pids[1] or pids[0] & pids[2]
                         or pids[1] & pids[2])

    def test_mixed_root_requires_explicit_camera_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            self._build_v2_tree(directory)
            with self.assertRaisesRegex(ValueError, "must both be given"):
                build_records(namespace(directory, directory))

    def test_unassigned_camera_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self._build_v2_tree(directory)
            with self.assertRaisesRegex(ValueError, "assigned to no domain"):
                build_records(namespace(
                    directory, directory, V2_AERIAL,
                    "c02;c12,c13;c22,c23"))

    def test_separate_roots_keep_v1_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            aerial = osp.join(directory, "aerial")
            ground = osp.join(directory, "ground")
            for angle in range(3):
                touch(aerial, "P0001_T0000{}_C{}0_F1.jpg".format(angle, angle))
                touch(ground, "P0001_T0000{}_C{}3_F1.jpg".format(angle, angle))
            records = build_records(namespace(
                aerial, ground, "c00,c10,c20", "c03;c13;c23"))
        self.assertEqual(sorted({(r.domain, r.camera, r.fold) for r in records}),
                         [("aerial", "c00", 0), ("aerial", "c10", 1),
                          ("aerial", "c20", 2), ("ground", "c03", 0),
                          ("ground", "c13", 1), ("ground", "c23", 2)])


class PerCameraGroupingTests(unittest.TestCase):
    def test_near_ratio_is_applied_inside_each_altitude_camera(self):
        rows = []
        for camera in ("c00", "c10", "c20"):
            for index in range(10):
                rows.append({"image_path": "{}_{}".format(camera, index),
                             "camera": camera,
                             "p_ground": index / 10.0})
        output, _ = add_view_groups(rows, 0.5)
        for camera in ("c00", "c10", "c20"):
            near = [row for row in output if row["camera"] == camera
                    and row["view_group_per_camera"] == "A_near"]
            self.assertEqual(len(near), 5)


class PreflightTests(unittest.TestCase):
    def _tree(self, directory):
        outer = Path(directory) / "AG-ReID.v2"
        train = outer / "AG-ReID.v2" / "train_all"
        masks = Path(directory) / "AG-ReID.v2_SCHP" / "train_all"
        names = ["0001/P0001T00001A0C0F1.jpg", "0001/P0001T00001A0C3F2.jpg",
                 "0002/P0002T00002A1C0F1.jpg"]
        for name in names:
            touch(train, name)
        return outer, train, masks, names

    def test_masks_complete_and_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, _, masks, names = self._tree(directory)
            for name in names[:2]:
                touch(masks, name[:-4] + ".png")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                inspect_masks(str(outer), str(masks))
            touch(masks, names[2][:-4] + ".png")
            inspect_masks(str(outer), str(masks))

    def test_view_scores_must_cover_every_aerial_image_once(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, train, _, _ = self._tree(directory)
            score = Path(directory) / "scores.csv"
            header = "image_path,pid,camera,p_ground_foreground,view_group\n"
            score.write_text(
                header + str(train / "0001/P0001T00001A0C0F1.jpg")
                + ",10000010,c00,0.4,A_far\n")
            with self.assertRaisesRegex(RuntimeError, "missing=1"):
                inspect_view_scores(str(outer), str(score))
            score.write_text(
                header + str(train / "0001/P0001T00001A0C0F1.jpg")
                + ",10000010,c00,0.4,A_far\n"
                + str(train / "0002/P0002T00002A1C0F1.jpg")
                + ",20000021,c10,0.6,A_near\n")
            inspect_view_scores(str(outer), str(score))


if __name__ == "__main__":
    unittest.main()
