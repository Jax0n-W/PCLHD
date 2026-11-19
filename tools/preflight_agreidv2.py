#!/usr/bin/env python3
"""CPU-only AG-ReID.v2 data and protocol preflight."""
from __future__ import absolute_import, print_function

import argparse
import csv
import glob
import os.path as osp
import sys
from collections import Counter, defaultdict


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agreid_identity import (  # noqa: E402
    DATASET_SPECS, parse_agreid_name, parse_agreidv2_manifest,
    resolve_agreidv2_paths, validate_agreidv2_protocol)


def inspect_train(data_dir):
    _, inner_root = resolve_agreidv2_paths(data_dir)
    spec = DATASET_SPECS["agreidv2"]
    train_root = osp.join(inner_root, spec["train_dir"])
    if not osp.isdir(train_root):
        raise FileNotFoundError(
            "AG-ReID.v2 train_all not found: {}".format(train_root))
    paths = sorted(glob.glob(
        osp.join(train_root, "**", "*.jpg"), recursive=True))
    parsed = [parse_agreid_name(osp.basename(path)) for path in paths]
    camera_counts = Counter(item.camid for item in parsed)
    identity_count = len({item.pid for item in parsed})
    aerial_count = camera_counts.get(0, 0)
    ground_count = camera_counts.get(2, 0) + camera_counts.get(3, 0)
    print("train_all images: {}".format(len(paths)))
    print("train_all camera counts: {}".format(
        dict(sorted(camera_counts.items()))))
    print("train_all identities (P/T/A): {}".format(identity_count))
    print("training domains: aerial(C0)={} ground(C2+C3)={}".format(
        aerial_count, ground_count))
    if (len(paths) != spec["train_images"]
            or dict(camera_counts) != spec["camera_images"]
            or identity_count != spec["train_identities"]
            or aerial_count != 21217
            or ground_count != 30313):
        raise RuntimeError(
            "AG-ReID.v2 train_all mismatch: expected images={}, cameras={}, "
            "identities=807, aerial=21217, ground=30313; actual images={}, "
            "cameras={}, identities={}, aerial={}, ground={}".format(
                spec["train_images"], spec["camera_images"], len(paths),
                dict(sorted(camera_counts.items())), identity_count,
                aerial_count, ground_count))


def _train_paths(data_dir):
    _, inner_root = resolve_agreidv2_paths(data_dir)
    train_root = osp.join(inner_root, DATASET_SPECS["agreidv2"]["train_dir"])
    paths = sorted(glob.glob(
        osp.join(train_root, "**", "*.jpg"), recursive=True))
    return train_root, paths


def inspect_masks(data_dir, mask_dir):
    """Every train image must have <mask_dir>/<relative path>.png."""
    train_root, paths = _train_paths(data_dir)
    mask_dir = osp.abspath(osp.expanduser(mask_dir))
    if not osp.isdir(mask_dir):
        raise FileNotFoundError("SCHP mask dir not found: {}".format(mask_dir))
    found = Counter()
    total = Counter()
    first_missing = None
    for path in paths:
        camera = parse_agreid_name(osp.basename(path)).camid
        total[camera] += 1
        relative = osp.relpath(path, train_root)
        mask = osp.splitext(osp.join(mask_dir, relative))[0] + ".png"
        if osp.isfile(mask):
            found[camera] += 1
        elif first_missing is None:
            first_missing = mask
    for camera in sorted(total):
        print("SCHP masks C{}: {}/{}".format(
            camera, found[camera], total[camera]))
    if first_missing is not None:
        raise RuntimeError(
            "SCHP masks incomplete ({}/{}); first missing: {}".format(
                sum(found.values()), sum(total.values()), first_missing))


def inspect_view_scores(data_dir, score_file, near_ratio=0.5):
    """Every C0 train image must appear exactly once in the view-score CSV."""
    _, paths = _train_paths(data_dir)
    aerial = {osp.basename(path) for path in paths
              if parse_agreid_name(osp.basename(path)).camid == 0}
    score_file = osp.abspath(osp.expanduser(score_file))
    if not osp.isfile(score_file):
        raise FileNotFoundError("view-score CSV not found: {}".format(score_file))
    seen = Counter()
    per_camera = defaultdict(int)
    with open(score_file, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = osp.basename(row["image_path"].replace("\\", "/"))
            seen[name] += 1
            per_camera[row["camera"]] += 1
    duplicates = [name for name, count in seen.items() if count > 1]
    missing = sorted(aerial - set(seen))
    extra = sorted(set(seen) - aerial)
    print("view scores: rows={} aerial_train={} per_camera={}".format(
        sum(seen.values()), len(aerial), dict(sorted(per_camera.items()))))
    print("view scores: expected A_near per camera at ratio {}: {}".format(
        near_ratio, {camera: int(count * near_ratio + 0.999999)
                     for camera, count in sorted(per_camera.items())}))
    if duplicates or missing or extra:
        raise RuntimeError(
            "view-score CSV mismatch: duplicates={} missing={} extra={} "
            "(first missing: {})".format(
                len(duplicates), len(missing), len(extra),
                missing[0] if missing else None))


def inspect_protocols(data_dir):
    for protocol_key, spec in DATASET_SPECS["agreidv2"]["protocols"].items():
        query, gallery = parse_agreidv2_manifest(
            data_dir, protocol_key, validate=True)
        actual = validate_agreidv2_protocol(protocol_key, query, gallery)
        print(
            "{} {}: query={} gallery={} identities(q/g/union)={}/{}/{}"
            .format(
                protocol_key, spec["name"], actual["query_images"],
                actual["gallery_images"], actual["query_identities"],
                actual["gallery_identities"], actual["union_identities"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate AG-ReID.v2 without loading a model or GPU")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--mask-dir", default="",
        help="optional SCHP mask root mirroring train_all (D-AGVA/classifier)")
    parser.add_argument(
        "--view-score-file", default="",
        help="optional aerial view-score CSV for three-domain Stage 2")
    parser.add_argument("--view-near-ratio", type=float, default=0.5)
    args = parser.parse_args(argv)
    outer_root, inner_root = resolve_agreidv2_paths(args.data_dir)
    print("AG-ReID.v2 outer root: {}".format(outer_root))
    print("AG-ReID.v2 inner root: {}".format(inner_root))
    inspect_train(outer_root)
    inspect_protocols(outer_root)
    if args.mask_dir:
        inspect_masks(outer_root, args.mask_dir)
    if args.view_score_file:
        inspect_view_scores(outer_root, args.view_score_file,
                            args.view_near_ratio)
    print("AGREIDV2_PREFLIGHT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
