"""Training-only horizontal aspect-ratio jitter for AG-ReID."""

from __future__ import absolute_import

import math
import os
import os.path as osp
import random

from PIL import Image, ImageDraw

from ChannelAug import ChannelRandomErasing
from . import transforms as T


ASPECT_JITTER_MODES = ("off", "both", "view", "geometry")
ASPECT_JITTER_STAGES = (1, 2, 12)


def effective_aspect_max_ratio(args):
    """Return the mode-specific default without hiding an explicit value."""
    value = getattr(args, "aspect_jitter_max_ratio", None)
    if value is not None:
        return float(value)
    return 0.3 if getattr(args, "aspect_jitter", "off") == "geometry" else 0.2


def apply_horizontal_ratio(image, ratio):
    """Resize only width, then center-crop or zero-pad to original size."""
    if not isinstance(image, Image.Image):
        raise TypeError("Aspect jitter expects a PIL image")
    width, height = image.size
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("aspect ratio must be positive")
    new_width = max(1, int(round(width * ratio)))
    if new_width == width:
        return image.copy()
    resized = image.resize((new_width, height), Image.Resampling.BICUBIC)
    if new_width > width:
        left = (new_width - width) // 2
        return resized.crop((left, 0, left + width, height))
    left = (width - new_width) // 2
    canvas = Image.new(image.mode, (width, height), 0)
    canvas.paste(resized, (left, 0))
    return canvas


class AspectJitter(object):
    """Log-uniform fixed-range horizontal jitter."""

    DIRECTIONS = ("both", "narrow", "widen")

    def __init__(self, max_ratio=0.2, prob=0.5, direction="both"):
        self.max_ratio = float(max_ratio)
        self.prob = float(prob)
        self.direction = str(direction)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if not 0.0 <= self.max_ratio <= 1.0:
            raise ValueError("max_ratio must be in [0, 1]")
        if self.direction not in self.DIRECTIONS:
            raise ValueError("unknown aspect direction: {}".format(
                self.direction))
        self.last_ratio = 1.0

    def sample_ratio(self):
        limit = math.log1p(self.max_ratio)
        if self.direction == "narrow":
            low, high = -limit, 0.0
        elif self.direction == "widen":
            low, high = 0.0, limit
        else:
            low, high = -limit, limit
        return math.exp(random.uniform(low, high))

    def __call__(self, image):
        if self.prob <= 0.0 or random.random() >= self.prob:
            self.last_ratio = 1.0
            return image
        self.last_ratio = self.sample_ratio()
        return apply_horizontal_ratio(image, self.last_ratio)


def aspect_jitter_enabled(args, stage):
    mode = getattr(args, "aspect_jitter", "off")
    stages = int(getattr(args, "aspect_jitter_stages", 1))
    return mode != "off" and str(int(stage)) in str(stages)


def geometry_aspect_jitter_enabled(args, stage):
    return (aspect_jitter_enabled(args, stage)
            and getattr(args, "aspect_jitter", "off") == "geometry")


def fixed_aspect_direction(args, domain):
    mode = getattr(args, "aspect_jitter", "off")
    if mode == "both":
        return "both"
    if mode == "view":
        return "narrow" if domain == "aerial" else "widen"
    return None


def build_aspect_train_transform(args, stage, domain, height, width,
                                 normalizer, crop_size=None,
                                 extra_before_tensor=None):
    """Build an opt-in transform without changing legacy constructors."""
    crop_size = crop_size or (height, width)
    operations = [T.Resize((height, width), interpolation=3)]
    direction = (fixed_aspect_direction(args, domain)
                 if aspect_jitter_enabled(args, stage) else None)
    if direction is not None:
        operations.append(AspectJitter(
            max_ratio=effective_aspect_max_ratio(args),
            prob=args.aspect_jitter_prob, direction=direction))
    operations.extend([
        T.Pad(10), T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
    ])
    operations.extend(list(extra_before_tensor or ()))
    operations.extend([
        T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5),
    ])
    return T.Compose(operations)


def _absolute_sample_path(sample, root):
    path = sample[0] if isinstance(sample, (tuple, list)) else sample
    return path if osp.isabs(path) or not root else osp.join(root, path)


def _choose_samples(samples, count, seed):
    samples = list(samples)
    if not samples:
        return []
    rng = random.Random(seed)
    if len(samples) >= count:
        return rng.sample(samples, count)
    return [samples[index % len(samples)] for index in range(count)]


def save_aspect_preview(rows, output_path, panel_size=(144, 288)):
    """Save labeled lower/identity/upper panels for manual inspection."""
    rows = list(rows)
    if not rows:
        return None
    panel_width, panel_height = panel_size
    label_height = 22
    canvas = Image.new(
        "RGB", (panel_width * 3, (panel_height + label_height) * len(rows)),
        "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, image, ratios) in enumerate(rows):
        y = row_index * (panel_height + label_height)
        draw.text((4, y + 4), str(label), fill="black")
        for column, ratio in enumerate(ratios):
            transformed = apply_horizontal_ratio(image, ratio)
            transformed = transformed.resize(
                panel_size, Image.Resampling.BICUBIC)
            canvas.paste(transformed, (column * panel_width, y + label_height))
            draw.text(
                (column * panel_width + 3, y + label_height + 3),
                "r={:.3f}".format(ratio), fill="white",
                stroke_width=1, stroke_fill="black")
    directory = osp.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    canvas.save(output_path, quality=92)
    print("[AspectJitter] preview: {}".format(output_path))
    return output_path


def _fixed_preview_rows(args, aerial_samples, ground_samples,
                        aerial_root, ground_root):
    rows = []
    max_ratio = effective_aspect_max_ratio(args)
    for group_index, (group, samples, root, domain) in enumerate((
            ("aerial", aerial_samples, aerial_root, "aerial"),
            ("ground", ground_samples, ground_root, "ground"))):
        direction = fixed_aspect_direction(args, domain)
        if direction == "narrow":
            ratios = (1.0 / (1.0 + max_ratio), 1.0, 1.0)
        elif direction == "widen":
            ratios = (1.0, 1.0, 1.0 + max_ratio)
        else:
            ratios = (1.0 / (1.0 + max_ratio), 1.0,
                      1.0 + max_ratio)
        selected = _choose_samples(
            samples, 4, int(getattr(args, "seed", 1)) + group_index)
        for sample_index, sample in enumerate(selected):
            path = _absolute_sample_path(sample, root)
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            rows.append(("{} #{}".format(group, sample_index + 1),
                         image, ratios))
    return rows


def prepare_aspect_jitter(args, stage, aerial_samples, ground_samples,
                          aerial_root, ground_root, logs_dir):
    """Validate, print and prepare the active stage, including its preview."""
    if not aspect_jitter_enabled(args, stage):
        return None
    print("========== Aspect Jitter ==========")
    print("Stage: {}".format(stage))
    print("Mode: {}".format(args.aspect_jitter))
    print("Probability: {}".format(args.aspect_jitter_prob))
    print("Maximum ratio: {}".format(effective_aspect_max_ratio(args)))
    print("Enabled stages: {}".format(args.aspect_jitter_stages))
    preview_path = osp.join(logs_dir, "aspect_jitter_preview.jpg")
    if args.aspect_jitter == "geometry":
        from .aspect_geometry import prepare_geometry_aspect
        result = prepare_geometry_aspect(
            args, aerial_samples, ground_samples, aerial_root, ground_root,
            preview_path)
    else:
        rows = _fixed_preview_rows(
            args, aerial_samples, ground_samples, aerial_root, ground_root)
        save_aspect_preview(rows, preview_path)
        result = None
    print("===================================")
    return result


def add_aspect_jitter_arguments(parser):
    parser.add_argument(
        "--aspect-jitter", choices=ASPECT_JITTER_MODES, default="off",
        help="training-only aspect jitter mode")
    parser.add_argument(
        "--aspect-jitter-max-ratio", type=float, default=None,
        help="default: 0.2 for fixed modes, 0.3 for geometry")
    parser.add_argument(
        "--aspect-jitter-prob", type=float, default=0.5)
    parser.add_argument(
        "--aspect-jitter-stages", type=int,
        choices=ASPECT_JITTER_STAGES, default=1)
    parser.add_argument(
        "--aspect-jitter-ground-mask-dir", default="",
        help="ground SCHP masks used by geometry mode")


def validate_aspect_jitter_args(args):
    if getattr(args, "aspect_jitter", "off") not in ASPECT_JITTER_MODES:
        raise ValueError("unknown --aspect-jitter mode")
    if not 0.0 <= float(args.aspect_jitter_prob) <= 1.0:
        raise ValueError("--aspect-jitter-prob must be in [0, 1]")
    max_ratio = effective_aspect_max_ratio(args)
    if not 0.0 <= max_ratio <= 1.0:
        raise ValueError("--aspect-jitter-max-ratio must be in [0, 1]")
    args.aspect_jitter_max_ratio = max_ratio
    if int(args.aspect_jitter_stages) not in ASPECT_JITTER_STAGES:
        raise ValueError("--aspect-jitter-stages must be 1, 2, or 12")
