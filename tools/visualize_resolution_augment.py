#!/usr/bin/env python3
from __future__ import absolute_import, print_function

"""Create an original/global/single/multi/blur training-image contact sheet."""

import argparse
import os.path as osp
import random
import sys

from PIL import Image, ImageDraw, ImageOps


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustercontrast import datasets  # noqa: E402
from clustercontrast.utils.data.resolution_augment import (  # noqa: E402
    CrossDomainResolutionAugment, build_height_distribution)


DATASET_NAMES = {
    "agreid": ("agreid_ir", "agreid_rgb"),
    "agreidv2": ("agreidv2_ir", "agreidv2_rgb"),
}
MODES = ("original", "global", "single", "multi", "blur")


def _load_dataset(name, data_dir, trial):
    return datasets.create(
        name, osp.join(data_dir, name), trial=trial, verbose=False)


def _preview(image, size):
    canvas = Image.new("RGB", size, "white")
    contained = ImageOps.contain(image.convert("RGB"), size)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def create_contact_sheet(data_dir, dataset_key, output, trial=1,
                         samples_per_domain=8, seed=1):
    aerial_name, ground_name = DATASET_NAMES[dataset_key]
    aerial = _load_dataset(aerial_name, data_dir, trial)
    ground = _load_dataset(ground_name, data_dir, trial)
    aerial_train = sorted(aerial.train)
    ground_train = sorted(ground.train)
    if min(len(aerial_train), len(ground_train)) < samples_per_domain:
        raise RuntimeError("each domain needs at least {} training images".format(
            samples_per_domain))

    aerial_heights = build_height_distribution(aerial_train)
    ground_heights = build_height_distribution(ground_train)
    selection_rng = random.Random(seed)
    chosen = [
        ("aerial", sample, ground_heights)
        for sample in selection_rng.sample(aerial_train, samples_per_domain)
    ] + [
        ("ground", sample, aerial_heights)
        for sample in selection_rng.sample(ground_train, samples_per_domain)
    ]

    cell_size = (160, 300)
    label_height = 22
    header_height = 32
    sheet = Image.new(
        "RGB", (cell_size[0] * len(MODES),
                header_height + (cell_size[1] + label_height) * len(chosen)),
        "white")
    draw = ImageDraw.Draw(sheet)
    for column, mode in enumerate(MODES):
        draw.text((column * cell_size[0] + 5, 8), mode, fill="black")

    for row, (domain, sample, opposite_heights) in enumerate(chosen):
        path = sample[0]
        with Image.open(path) as source:
            original = source.convert("RGB")
        images = [original]
        for mode_index, mode in enumerate(MODES[1:], start=1):
            augmenter = CrossDomainResolutionAugment(
                opposite_heights, mode=mode, probability=1.0)
            images.append(augmenter(
                original, rng=random.Random(
                    seed * 100000 + row * 100 + mode_index)))

        y = header_height + row * (cell_size[1] + label_height)
        label = "{} {:02d} h={} {}".format(
            domain, row % samples_per_domain + 1,
            original.height, osp.basename(path))
        draw.text((5, y), label[:100], fill="black")
        y += label_height
        for column, image in enumerate(images):
            sheet.paste(_preview(image, cell_size),
                        (column * cell_size[0], y))

    output = osp.abspath(osp.expanduser(output))
    sheet.save(output)
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize cross-domain resolution augmentation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dataset-key", choices=tuple(sorted(DATASET_NAMES)),
        default="agreid")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output", default=osp.join(ROOT, "resolution_augment_preview.png"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = create_contact_sheet(
        osp.abspath(osp.expanduser(args.data_dir)), args.dataset_key,
        args.output, trial=args.trial,
        samples_per_domain=args.samples_per_domain, seed=args.seed)
    print("Resolution augmentation preview: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
