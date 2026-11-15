from __future__ import absolute_import, print_function

"""Cross-domain, resolution-matching augmentation for AG-ReID training.

The image operations in this module deliberately use Python's global
``random`` module by default.  PyTorch seeds that module independently in
every DataLoader worker, so the augmentation follows the worker RNG stream
and remains reproducible under the existing ``--seed`` setup.
"""

import math
import multiprocessing
import os.path as osp
import random

import numpy as np
from PIL import Image, ImageFilter


RES_AUG_MODES = ("off", "global", "single", "multi", "blur")
RES_AUG_DOMAINS = ("auto", "aerial", "ground", "both")
RES_AUG_STAGES = ("1", "2", "both")
DEFAULT_PROBABILITY = 0.5
DEFAULT_MIN_SCALE = 0.15
DEFAULT_PATCHES = (3, 5)
DEFAULT_PATCH_AREA = (0.05, 0.20)
HEIGHT_PERCENTILES = (5, 25, 50, 75, 95)

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    BICUBIC = Image.BICUBIC


def _float_text(value):
    return "{:g}".format(float(value))


def add_resolution_augment_arguments(parser):
    group = parser.add_argument_group("cross-domain resolution augmentation")
    group.add_argument(
        "--res-aug-mode", choices=RES_AUG_MODES, default="off")
    group.add_argument(
        "--res-aug-prob", type=float, default=DEFAULT_PROBABILITY)
    group.add_argument(
        "--res-aug-domains", choices=RES_AUG_DOMAINS, default="auto")
    group.add_argument(
        "--res-aug-stages", choices=RES_AUG_STAGES, default="both")
    group.add_argument(
        "--res-aug-min-scale", type=float, default=DEFAULT_MIN_SCALE)
    group.add_argument(
        "--res-aug-patches", type=int, nargs=2,
        metavar=("MIN", "MAX"), default=DEFAULT_PATCHES)
    group.add_argument(
        "--res-aug-patch-area", type=float, nargs=2,
        metavar=("MIN", "MAX"), default=DEFAULT_PATCH_AREA)
    return parser


def validate_resolution_augment_args(args):
    probability = float(args.res_aug_prob)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("--res-aug-prob must be in [0, 1]")
    min_scale = float(args.res_aug_min_scale)
    if not 0.0 < min_scale < 1.0:
        raise ValueError("--res-aug-min-scale must be in (0, 1)")
    patch_min, patch_max = (int(value) for value in args.res_aug_patches)
    if patch_min < 1 or patch_min > patch_max:
        raise ValueError(
            "--res-aug-patches requires positive MIN <= MAX")
    area_min, area_max = (float(value)
                          for value in args.res_aug_patch_area)
    if not 0.0 < area_min <= area_max < 1.0:
        raise ValueError(
            "--res-aug-patch-area requires 0 < MIN <= MAX < 1")


def resolution_log_suffix(args):
    if getattr(args, "res_aug_mode", "off") == "off":
        return ""
    suffix = "_res{}_p{}".format(
        args.res_aug_mode, _float_text(args.res_aug_prob))
    if args.res_aug_domains != "auto":
        suffix += "_dom{}".format(args.res_aug_domains)
    if str(args.res_aug_stages) != "both":
        suffix += "_st{}".format(args.res_aug_stages)
    if abs(float(args.res_aug_min_scale) - DEFAULT_MIN_SCALE) > 1e-12:
        suffix += "_min{}".format(_float_text(args.res_aug_min_scale))
    patch_range = tuple(int(value) for value in args.res_aug_patches)
    if patch_range != DEFAULT_PATCHES:
        suffix += "_k{}-{}".format(*patch_range)
    area_range = tuple(float(value) for value in args.res_aug_patch_area)
    if any(abs(value - default) > 1e-12 for value, default in zip(
            area_range, DEFAULT_PATCH_AREA)):
        suffix += "_a{}-{}".format(
            _float_text(area_range[0]), _float_text(area_range[1]))
    return suffix


def append_resolution_log_suffix(name, args):
    suffix = resolution_log_suffix(args)
    if not suffix or suffix in str(name):
        return name
    return "{}{}".format(name, suffix)


def resolution_enabled_for_stage(args, stage):
    if getattr(args, "res_aug_mode", "off") == "off":
        return False
    configured = str(args.res_aug_stages)
    return configured == "both" or configured == str(int(stage))


def resolution_enabled_for_domain(args, domain):
    configured = getattr(args, "res_aug_domains", "auto")
    return configured in ("auto", "both", str(domain))


def sample_path(sample, root=None):
    path = sample[0] if isinstance(sample, (tuple, list)) else sample
    path = osp.expanduser(str(path))
    if root is not None and not osp.isabs(path):
        path = osp.join(root, path)
    return osp.abspath(path)


def read_original_height(path):
    """Read only the encoded image dimensions and return its height."""
    with Image.open(path) as image:
        return int(image.height)


def build_height_distribution(samples, root=None):
    """Build an empirical height distribution from exactly ``samples``."""
    return tuple(read_original_height(sample_path(sample, root))
                 for sample in samples)


def height_percentiles(heights, percentiles=HEIGHT_PERCENTILES):
    if not heights:
        raise ValueError("height distribution cannot be empty")
    values = np.percentile(np.asarray(heights, dtype=np.float64), percentiles)
    return tuple(float(value) for value in values)


def format_height_percentiles(domain, heights):
    values = height_percentiles(heights)
    fields = ["q{:02d}={:.1f}".format(percentile, value)
              for percentile, value in zip(HEIGHT_PERCENTILES, values)]
    return "[RES-AUG] train_heights domain={} {}".format(
        domain, " ".join(fields))


def sample_matched_scale(source_height, target_heights, min_scale,
                         rng=random):
    if int(source_height) <= 0:
        raise ValueError("source height must be positive")
    if not target_heights:
        raise ValueError("target height distribution cannot be empty")
    target_height = int(rng.choice(target_heights))
    raw_scale = float(target_height) / float(source_height)
    if raw_scale >= 1.0:
        return 1.0, raw_scale, False
    return max(float(min_scale), raw_scale), raw_scale, True


def degrade_resolution(image, scale):
    """Bicubic downsample and restore an entire PIL image."""
    if float(scale) >= 1.0:
        return image
    width, height = image.size
    low_width = max(1, int(round(width * float(scale))))
    low_height = max(1, int(round(height * float(scale))))
    low = image.resize((low_width, low_height), BICUBIC)
    return low.resize((width, height), BICUBIC)


def degrade_region(image, rectangle, scale):
    if float(scale) >= 1.0:
        return image
    left, top, right, bottom = rectangle
    patch = image.crop((left, top, right, bottom))
    degraded = degrade_resolution(patch, scale)
    output = image.copy()
    output.paste(degraded, (left, top, right, bottom))
    return output


def rectangles_overlap(first, second):
    return not (first[2] <= second[0] or second[2] <= first[0]
                or first[3] <= second[1] or second[3] <= first[1])


def sample_non_overlapping_rectangles(
        image_size, count_range=DEFAULT_PATCHES,
        area_range=DEFAULT_PATCH_AREA, rng=random, max_attempts=100):
    """Return (rectangles, requested_count, fallback_count).

    Coordinates use PIL's ``(left, top, right, bottom)`` convention.  A
    shared rejection budget of ``max_attempts`` is used for the whole image;
    any unplaced requested rectangles are counted as fallbacks.
    """
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    count_min, count_max = (int(value) for value in count_range)
    requested = rng.randint(count_min, count_max)
    rectangles = []
    attempts = 0
    full_area = float(width * height)
    while len(rectangles) < requested and attempts < int(max_attempts):
        attempts += 1
        target_area = rng.uniform(*area_range) * full_area
        aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        patch_width = max(1, int(round(math.sqrt(target_area * aspect))))
        patch_height = max(1, int(round(math.sqrt(target_area / aspect))))
        if patch_width > width or patch_height > height:
            continue
        left = rng.randint(0, width - patch_width)
        top = rng.randint(0, height - patch_height)
        candidate = (
            left, top, left + patch_width, top + patch_height)
        if any(rectangles_overlap(candidate, placed)
               for placed in rectangles):
            continue
        rectangles.append(candidate)
    return rectangles, requested, requested - len(rectangles)


class ResolutionAugmentStats(object):
    """Small process-shared accumulator used by DataLoader workers."""

    # seen, applied, downscaled, scale_sum, scale_count, patches, fallbacks
    _SIZE = 7

    def __init__(self):
        self._values = multiprocessing.Array("d", self._SIZE, lock=True)

    def reset(self):
        with self._values.get_lock():
            for index in range(self._SIZE):
                self._values[index] = 0.0

    def record(self, applied=False, downscaled=False, scales=(),
               patches_placed=0, patch_fallbacks=0):
        scales = tuple(float(value) for value in scales)
        with self._values.get_lock():
            self._values[0] += 1.0
            self._values[1] += float(bool(applied))
            self._values[2] += float(bool(downscaled))
            self._values[3] += sum(scales)
            self._values[4] += len(scales)
            self._values[5] += int(patches_placed)
            self._values[6] += int(patch_fallbacks)

    def snapshot(self):
        with self._values.get_lock():
            values = tuple(float(value) for value in self._values)
        seen, applied, downscaled, scale_sum, scale_count, patches, fallbacks = values
        return {
            "seen": int(seen),
            "applied": (applied / seen) if seen else 0.0,
            "downscaled": (downscaled / seen) if seen else 0.0,
            "mean_scale": (scale_sum / scale_count)
            if scale_count else 1.0,
            "patches_placed_mean": (patches / applied)
            if applied else 0.0,
            "patch_fallbacks": int(fallbacks),
        }


class CrossDomainResolutionAugment(object):
    """Serializable PIL pre-transform using an opposite-domain distribution."""

    def __init__(self, target_heights, mode="global",
                 probability=DEFAULT_PROBABILITY,
                 min_scale=DEFAULT_MIN_SCALE,
                 patches=DEFAULT_PATCHES,
                 patch_area=DEFAULT_PATCH_AREA, stats=None):
        if mode not in RES_AUG_MODES or mode == "off":
            raise ValueError("active augmentation requires a non-off mode")
        self.target_heights = tuple(int(value) for value in target_heights)
        self.mode = mode
        self.probability = float(probability)
        self.min_scale = float(min_scale)
        self.patches = tuple(int(value) for value in patches)
        self.patch_area = tuple(float(value) for value in patch_area)
        self.stats = stats if stats is not None else ResolutionAugmentStats()
        if not self.target_heights:
            raise ValueError("target height distribution cannot be empty")

    def reset_stats(self):
        self.stats.reset()

    def snapshot(self):
        return self.stats.snapshot()

    def _scale(self, image, rng):
        return sample_matched_scale(
            image.height, self.target_heights, self.min_scale, rng=rng)

    def __call__(self, image, path=None, rng=None):
        del path  # accepted for Preprocessor's path-aware hook
        rng = random if rng is None else rng
        if rng.random() >= self.probability:
            self.stats.record()
            return image

        if self.mode in ("global", "blur"):
            scale, raw_scale, downscaled = self._scale(image, rng)
            del raw_scale
            if not downscaled:
                self.stats.record(applied=True)
                return image
            if self.mode == "global":
                output = degrade_resolution(image, scale)
            else:
                # Radius grows as the matched resolution shrinks.  The 0.5/s
                # conversion is intentionally simple and fixed for this
                # resolution-vs-general-blur control.
                output = image.filter(ImageFilter.GaussianBlur(
                    radius=0.5 / float(scale)))
            self.stats.record(
                applied=True, downscaled=True, scales=(scale,))
            return output

        if self.mode == "single":
            rectangles, requested, fallbacks = \
                sample_non_overlapping_rectangles(
                    image.size, count_range=(1, 1),
                    area_range=self.patch_area, rng=rng)
            del requested
            if not rectangles:
                self.stats.record(
                    applied=True, patch_fallbacks=fallbacks)
                return image
            scale, raw_scale, downscaled = self._scale(image, rng)
            del raw_scale
            output = (degrade_region(image, rectangles[0], scale)
                      if downscaled else image)
            self.stats.record(
                applied=True, downscaled=downscaled,
                scales=(scale,) if downscaled else (),
                patches_placed=1, patch_fallbacks=fallbacks)
            return output

        rectangles, requested, fallbacks = sample_non_overlapping_rectangles(
            image.size, count_range=self.patches,
            area_range=self.patch_area, rng=rng)
        del requested
        output = image
        scales = []
        for rectangle in rectangles:
            scale, raw_scale, downscaled = self._scale(image, rng)
            del raw_scale
            if downscaled:
                output = degrade_region(output, rectangle, scale)
                scales.append(scale)
        self.stats.record(
            applied=True, downscaled=bool(scales), scales=scales,
            patches_placed=len(rectangles), patch_fallbacks=fallbacks)
        return output


def build_resolution_augmenters(args, aerial_samples, ground_samples,
                                stage, aerial_root=None, ground_root=None,
                                printer=print):
    """Read train-only distributions and build per-domain preprocessors."""
    if not resolution_enabled_for_stage(args, stage):
        return {"aerial": None, "ground": None}
    aerial_heights = build_height_distribution(aerial_samples, aerial_root)
    ground_heights = build_height_distribution(ground_samples, ground_root)
    printer(format_height_percentiles("aerial", aerial_heights))
    printer(format_height_percentiles("ground", ground_heights))

    common = {
        "mode": args.res_aug_mode,
        "probability": args.res_aug_prob,
        "min_scale": args.res_aug_min_scale,
        "patches": args.res_aug_patches,
        "patch_area": args.res_aug_patch_area,
    }
    return {
        "aerial": CrossDomainResolutionAugment(
            ground_heights, **common)
        if resolution_enabled_for_domain(args, "aerial") else None,
        "ground": CrossDomainResolutionAugment(
            aerial_heights, **common)
        if resolution_enabled_for_domain(args, "ground") else None,
    }


def reset_resolution_stats(augmenters):
    for augmenter in augmenters.values():
        if augmenter is not None:
            augmenter.reset_stats()


def print_resolution_stats(stage, epoch, augmenters, printer=print):
    for domain in ("aerial", "ground"):
        augmenter = augmenters.get(domain)
        if augmenter is None:
            continue
        values = augmenter.snapshot()
        printer(
            "[RES-AUG] stage={} epoch={} domain={} applied={:.2f} "
            "downscaled={:.2f} mean_scale={:.2f} "
            "patches_placed_mean={:.1f} patch_fallbacks={}".format(
                int(stage), int(epoch), domain, values["applied"],
                values["downscaled"], values["mean_scale"],
                values["patches_placed_mean"],
                values["patch_fallbacks"]))
