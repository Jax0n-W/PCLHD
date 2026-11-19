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

from clustercontrast.utils.data.dynamic_agva import (  # noqa: E402
    report_dynamic_agva_mask_coverage)


def _args(mask_dir, image_root, strict):
    return argparse.Namespace(
        use_dynamic_agva=True, dynamic_agva_mask_dir=str(mask_dir),
        dynamic_agva_image_root=str(image_root), dynamic_agva_slots=1,
        dynamic_agva_min_displacement=0.03,
        dynamic_agva_max_displacement=0.04,
        dynamic_agva_strict_masks=strict)


class DynamicAGVACoverageTests(unittest.TestCase):
    def _tree(self, directory):
        images = Path(directory) / "AG-ReID.v2" / "train_all"
        masks = Path(directory) / "AG-ReID.v2_SCHP" / "train_all"
        names = ["0001/P0001T00001A0C0F1.jpg", "0002/P0002T00002A1C0F1.jpg"]
        for name in names:
            (images / name).parent.mkdir(parents=True, exist_ok=True)
            (images / name).write_bytes(b"")
        (masks / "0001").mkdir(parents=True)
        (masks / "0001" / "P0001T00001A0C0F1.png").write_bytes(b"")
        samples = [(str(images / name), -1, 0) for name in names]
        return images, masks, samples

    def test_mirrored_v2_masks_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            images, masks, samples = self._tree(directory)
            found, total, missing = report_dynamic_agva_mask_coverage(
                _args(masks, images, False), samples, None,
                printer=lambda *_: None)
        self.assertEqual((found, total), (1, 2))
        self.assertTrue(missing.endswith("P0002T00002A1C0F1.jpg"))

    def test_strict_masks_fail_on_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            images, masks, samples = self._tree(directory)
            with self.assertRaises(FileNotFoundError):
                report_dynamic_agva_mask_coverage(
                    _args(masks, images, True), samples, None,
                    printer=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
