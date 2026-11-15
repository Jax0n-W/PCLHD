#!/usr/bin/env python3
from __future__ import absolute_import, print_function

"""Create a deterministic contact sheet for manual augmentation inspection.

This module intentionally builds a preview-only sampling plan instead of
calling the training transform independently for every column.  That keeps
target heights, scales, and rectangles comparable within each row without
changing any training code.
"""

import argparse
import os
import os.path as osp
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clustercontrast import datasets  # noqa: E402
from clustercontrast.utils.data.resolution_augment import (  # noqa: E402
    DEFAULT_MIN_SCALE, DEFAULT_PATCHES, DEFAULT_PATCH_AREA,
    build_height_distribution, degrade_region, degrade_resolution,
    sample_matched_scale, sample_non_overlapping_rectangles)


DATASET_NAMES = {
    "agreid": ("agreid_ir", "agreid_rgb"),
    "agreidv2": ("agreidv2_ir", "agreidv2_rgb"),
}
MODES = ("original", "global", "single", "multi", "blur", "zoomed local")
PREVIEW_PROBABILITY = 1.0

try:
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9.1
    NEAREST = Image.NEAREST


def _load_dataset(name, data_dir, trial):
    return datasets.create(
        name, osp.join(data_dir, name), trial=trial, verbose=False)


def _load_font(size):
    candidates = (
        osp.join(os.environ.get("WINDIR", "C:\\Windows"),
                 "Fonts", "msyh.ttc"),
        osp.join(os.environ.get("WINDIR", "C:\\Windows"),
                 "Fonts", "simhei.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for candidate in candidates:
        if osp.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_text(draw, position, text, fill, font):
    """Draw Unicode when available and remain usable with a bitmap fallback."""
    try:
        draw.text(position, text, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(position, text.encode("ascii", "replace").decode("ascii"),
                  fill=fill, font=font)


def _preview(image, size, rectangles=()):
    """Fit an image and draw mapped rectangle outlines on the preview only."""
    canvas = Image.new("RGB", size, "white")
    contained = ImageOps.contain(image.convert("RGB"), size)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    if rectangles:
        draw = ImageDraw.Draw(canvas)
        scale_x = contained.width / float(image.width)
        scale_y = contained.height / float(image.height)
        for rect_left, rect_top, rect_right, rect_bottom in rectangles:
            mapped = (
                left + int(round(rect_left * scale_x)),
                top + int(round(rect_top * scale_y)),
                min(size[0] - 1,
                    left + int(round(rect_right * scale_x)) - 1),
                min(size[1] - 1,
                    top + int(round(rect_bottom * scale_y)) - 1),
            )
            draw.rectangle(mapped, outline=(255, 40, 40), width=1)
    return canvas


def _sample_scale(source_height, target_heights, rng):
    scale, raw_scale, downscaled = sample_matched_scale(
        source_height, target_heights, DEFAULT_MIN_SCALE, rng=rng)
    return {
        "target_height": int(round(raw_scale * source_height)),
        "scale": float(scale),
        "raw_scale": float(raw_scale),
        "downscaled": bool(downscaled),
    }


def _build_row_plan(image, target_heights, row_seed):
    """Sample exactly once, then reuse the complete plan across the row."""
    rng = random.Random(row_seed)
    common_scale = _sample_scale(image.height, target_heights, rng)
    rectangles, requested, fallbacks = sample_non_overlapping_rectangles(
        image.size, count_range=DEFAULT_PATCHES,
        area_range=DEFAULT_PATCH_AREA, rng=rng)
    multi_scales = []
    for index in range(len(rectangles)):
        # The first multi patch is the single patch, so it shares the common
        # target height and scale with global/blur/single as well.
        multi_scales.append(
            common_scale if index == 0 else
            _sample_scale(image.height, target_heights, rng))
    return {
        "seed": int(row_seed),
        "probability": PREVIEW_PROBABILITY,
        "common_scale": common_scale,
        "single_rectangles": tuple(rectangles[:1]),
        "multi_rectangles": tuple(rectangles),
        "multi_scales": tuple(multi_scales),
        "requested_rectangles": int(requested),
        "patch_fallbacks": int(fallbacks),
    }


def _apply_row_plan(image, plan):
    common = plan["common_scale"]
    if common["downscaled"]:
        global_image = degrade_resolution(image, common["scale"])
        blur_image = image.filter(ImageFilter.GaussianBlur(
            radius=0.5 / common["scale"]))
    else:
        global_image = image
        blur_image = image

    single_image = image
    if plan["single_rectangles"] and common["downscaled"]:
        single_image = degrade_region(
            image, plan["single_rectangles"][0], common["scale"])

    multi_image = image
    for rectangle, scale_info in zip(
            plan["multi_rectangles"], plan["multi_scales"]):
        if scale_info["downscaled"]:
            multi_image = degrade_region(
                multi_image, rectangle, scale_info["scale"])

    return {
        "original": image,
        "global": global_image,
        "single": single_image,
        "multi": multi_image,
        "blur": blur_image,
    }


def _scale_text(scale_info):
    if not scale_info["downscaled"]:
        return "s>=1 跳过"
    if abs(scale_info["scale"] - scale_info["raw_scale"]) > 1e-12:
        return "s={:.3f}（下限截断）".format(scale_info["scale"])
    return "s={:.3f}".format(scale_info["scale"])


def _annotation_lines(mode, plan, zoom_scale=None):
    if mode == "original":
        return ("s=原图", "矩形数量: 0")
    if mode == "multi":
        scale_labels = [
            _scale_text(scale_info) for scale_info in plan["multi_scales"]]
        split = (len(scale_labels) + 1) // 2
        return (" / ".join(scale_labels[:split]),
                " / ".join(scale_labels[split:]),
                "矩形数量: {}".format(len(plan["multi_rectangles"])))
    if mode == "single":
        return (_scale_text(plan["common_scale"]),
                "矩形数量: {}".format(len(plan["single_rectangles"])))
    if mode == "zoomed local":
        return (_scale_text(zoom_scale), "矩形数量: 1")
    return (_scale_text(plan["common_scale"]), "矩形数量: 0")


def _zoom_crop(image, rectangle, panel_size):
    """Crop inside one planned patch and render it at exactly 3x."""
    left, top, right, bottom = rectangle
    crop_width = max(1, min(right - left, panel_size[0] // 3))
    crop_height = max(1, min(bottom - top, panel_size[1] // 3))
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    crop_left = max(left, min(right - crop_width, center_x - crop_width // 2))
    crop_top = max(top, min(bottom - crop_height,
                            center_y - crop_height // 2))
    crop = image.crop((crop_left, crop_top,
                       crop_left + crop_width, crop_top + crop_height))
    return crop.resize((crop_width * 3, crop_height * 3), NEAREST)


def _zoom_comparison(original, processed, rectangle, size, font):
    """Show original and degraded crops in one column, both enlarged 3x."""
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    label_height = 18
    panel_height = size[1] // 2
    usable = (size[0], panel_height - label_height)
    for index, (name, image) in enumerate(
            (("original x3", original), ("degraded x3", processed))):
        y = index * panel_height
        _draw_text(draw, (4, y + 2), name, "black", font)
        zoomed = _zoom_crop(image, rectangle, usable)
        x = (size[0] - zoomed.width) // 2
        image_y = y + label_height + (usable[1] - zoomed.height) // 2
        canvas.paste(zoomed, (x, image_y))
    return canvas


def _select_zoom_source(original, processed, plan):
    """Prefer the shared single patch, then any actually degraded multi patch."""
    if (plan["single_rectangles"]
            and plan["common_scale"]["downscaled"]):
        return (processed["single"], plan["single_rectangles"][0],
                plan["common_scale"])
    for rectangle, scale_info in zip(
            plan["multi_rectangles"], plan["multi_scales"]):
        if scale_info["downscaled"]:
            return processed["multi"], rectangle, scale_info
    if plan["multi_rectangles"]:
        return (processed["multi"], plan["multi_rectangles"][0],
                plan["multi_scales"][0])
    # This is only reachable if the rejection sampler placed no rectangles.
    fallback = (0, 0, original.width, original.height)
    return processed["global"], fallback, plan["common_scale"]


def _render_cell(image, preview_size, footer_height, annotation,
                 font, rectangles=()):
    cell = Image.new(
        "RGB", (preview_size[0], preview_size[1] + footer_height), "white")
    cell.paste(_preview(image, preview_size, rectangles=rectangles), (0, 0))
    draw = ImageDraw.Draw(cell)
    draw.line((0, preview_size[1], preview_size[0], preview_size[1]),
              fill=(205, 205, 205), width=1)
    for line_index, line in enumerate(annotation):
        _draw_text(draw, (4, preview_size[1] + 3 + line_index * 19),
                   line, "black", font)
    return cell


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

    preview_size = (210, 300)
    footer_height = 62
    row_label_height = 24
    header_height = 54
    cell_size = (preview_size[0], preview_size[1] + footer_height)
    body_font = _load_font(13)
    small_font = _load_font(11)
    sheet = Image.new(
        "RGB", (cell_size[0] * len(MODES),
                header_height
                + (cell_size[1] + row_label_height) * len(chosen)),
        "white")
    draw = ImageDraw.Draw(sheet)
    _draw_text(
        draw, (5, 3), "人工检查预览：概率 p=1（强制施加）",
        "black", body_font)
    for column, mode in enumerate(MODES):
        _draw_text(draw, (column * cell_size[0] + 5, 29),
                   mode, "black", body_font)

    for row, (domain, sample, opposite_heights) in enumerate(chosen):
        path = sample[0]
        with Image.open(path) as source:
            original = source.convert("RGB")
        row_seed = seed * 100000 + row
        plan = _build_row_plan(original, opposite_heights, row_seed)
        processed = _apply_row_plan(original, plan)
        zoom_image, zoom_rectangle, zoom_scale = _select_zoom_source(
            original, processed, plan)

        y = header_height + row * (cell_size[1] + row_label_height)
        label = "{} {:02d} h={} seed={} {}".format(
            domain, row % samples_per_domain + 1,
            original.height, row_seed, osp.basename(path))
        _draw_text(draw, (5, y + 3), label[:130], "black", body_font)
        y += row_label_height

        for column, mode in enumerate(MODES):
            if mode == "zoomed local":
                preview = _zoom_comparison(
                    original, zoom_image, zoom_rectangle,
                    preview_size, small_font)
                cell = Image.new(
                    "RGB", (cell_size[0], cell_size[1]), "white")
                cell.paste(preview, (0, 0))
                footer = Image.new(
                    "RGB", (preview_size[0], footer_height), "white")
                footer_draw = ImageDraw.Draw(footer)
                annotation = _annotation_lines(mode, plan, zoom_scale)
                for line_index, line in enumerate(annotation):
                    _draw_text(footer_draw, (4, 3 + line_index * 19),
                               line, "black", small_font)
                cell.paste(footer, (0, preview_size[1]))
            else:
                rectangles = ()
                if mode == "single":
                    rectangles = plan["single_rectangles"]
                elif mode == "multi":
                    rectangles = plan["multi_rectangles"]
                cell = _render_cell(
                    processed[mode], preview_size, footer_height,
                    _annotation_lines(mode, plan), small_font,
                    rectangles=rectangles)
            sheet.paste(cell, (column * cell_size[0], y))

    output = osp.abspath(osp.expanduser(output))
    output_dir = osp.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
