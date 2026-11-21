"""D-AGVA sample flags and geometry-adaptive target selection."""

from __future__ import absolute_import

import os.path as osp
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from . import IterLoader
from .aerial_ground_augment import AerialGroundViewAugment
from .dynamic_agva import DynamicAGVAProcess
from .preprocessor import Preprocessor
from .sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from .schp_mask_loader import SCHPMaskLoader


class AerialGroundViewAugmentScaled(AerialGroundViewAugment):
    """Apply the original transform with a multiplicative strength scale."""

    def __call__(self, img, mask, strength_scale=1.0):
        original_range = self.strength_range
        scale = float(strength_scale)
        self.strength_range = tuple(value * scale for value in original_range)
        try:
            return super(AerialGroundViewAugmentScaled, self).__call__(img, mask)
        finally:
            self.strength_range = original_range


def mask_geometry_ratio(mask, image_size=None):
    """Return foreground bbox height/width using ``geom_gap.py`` rules."""
    if mask is None:
        return None
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask).astype(np.uint8))
    if image_size is not None and mask.size != tuple(image_size):
        mask = mask.resize(tuple(image_size), Image.Resampling.NEAREST)
    binary = np.asarray(mask) > 0
    if binary.ndim == 3:
        binary = binary[..., 0]
    if binary.ndim != 2 or not binary.any():
        return None
    height, width = binary.shape
    valid_rows = np.flatnonzero(
        binary.sum(axis=1) >= max(2.0, 0.02 * width))
    valid_cols = np.flatnonzero(
        binary.sum(axis=0) >= max(2.0, 0.02 * height))
    if valid_rows.size == 0 or valid_cols.size == 0:
        return None
    box_height = int(valid_rows[-1] - valid_rows[0] + 1)
    box_width = int(valid_cols[-1] - valid_cols[0] + 1)
    return float(box_height) / float(box_width)


def adaptive_strength_scale(own, reference, need_scale=0.20):
    """Compute clipped ground-target deformation strength."""
    own = float(own)
    reference = float(reference)
    need_scale = float(need_scale)
    if own <= 0 or need_scale <= 0:
        raise ValueError("own and need_scale must be positive")
    need = reference / own - 1.0
    return float(np.clip(need / need_scale, 0.0, 1.0))


class DynamicAGVAProcessWithFlag(DynamicAGVAProcess):
    """Return ``(image, really_applied)`` without changing fixed-mode RNG."""

    def __init__(self, *args, **kwargs):
        self.target = kwargs.pop("target", "fixed")
        self.need_scale = float(kwargs.pop("need_scale", 0.20))
        ground_ratios = kwargs.pop("ground_ratios", None)
        super(DynamicAGVAProcessWithFlag, self).__init__(*args, **kwargs)
        self.ground_ratios = (None if ground_ratios is None else
                              np.asarray(ground_ratios, dtype=np.float64))
        self._ground_ratio_choices = (
            None if self.ground_ratios is None
            else self.ground_ratios.tolist())
        # Debug changes only the return shape. It does not add random draws.
        self.augment = AerialGroundViewAugmentScaled(p=1.0, debug=True)

    def __call__(self, image, image_path):
        draw = random.randint(1, 3)
        if draw > self.apply_slots:
            return image, False

        if self.target == "fixed":
            # Keep these operations in precisely the original order.
            self.augment.max_displacement_ratio = random.uniform(
                *self.displacement_ratio_range)
            mask = self.mask_loader.load(
                image_path, img_shape=(image.height, image.width))
            result, info = self.augment(image, mask, strength_scale=1.0)
            return result, bool(info.applied)

        if self.target != "ground-mixture":
            raise ValueError("unknown D-AGVA target: {}".format(self.target))
        if self.ground_ratios is None or self.ground_ratios.size == 0:
            raise RuntimeError("ground-mixture target has no ground references")

        mask = self.mask_loader.load(
            image_path, img_shape=(image.height, image.width))
        own = mask_geometry_ratio(mask, image.size)
        if own is None:
            return image, False
        reference = float(random.choice(self._ground_ratio_choices))
        scale = adaptive_strength_scale(own, reference, self.need_scale)
        if scale <= 0.01:
            return image, False
        self.augment.max_displacement_ratio = random.uniform(
            *self.displacement_ratio_range)
        result, info = self.augment(image, mask, strength_scale=scale)
        return result, bool(info.applied)


class PreprocessorAugFlag(Preprocessor):
    """The standard preprocessor plus a boolean D-AGVA-applied field."""

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = osp.join(self.root, fname) if self.root is not None else fname
        img = Image.open(fpath).convert('RGB')
        is_aug = False
        if self.image_process is not None:
            img, is_aug = self.image_process(img, fpath)
        if self.resolution_process is not None:
            img = self.resolution_process(img, fpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, index, bool(is_aug)


def agva_flag_loader_enabled(args):
    return bool(
        getattr(args, 'dynamic_agva_no_memory_update', False)
        or getattr(args, 'dynamic_agva_target', 'fixed') != 'fixed')


def get_dynamic_agva_process_with_flag(args, image_root):
    if not getattr(args, 'use_dynamic_agva', False):
        return None
    effective_root = getattr(args, 'dynamic_agva_image_root', '') or image_root
    effective_root = osp.abspath(osp.expanduser(effective_root))
    cache_key = (
        effective_root, getattr(args, 'dynamic_agva_target', 'fixed'),
        float(getattr(args, 'dynamic_agva_need_scale', 0.20)))
    if (getattr(args, '_dynamic_agva_flag_process_key', None) == cache_key
            and getattr(args, '_dynamic_agva_flag_process', None) is not None):
        return args._dynamic_agva_flag_process
    process = DynamicAGVAProcessWithFlag(
        mask_dir=args.dynamic_agva_mask_dir,
        image_root=effective_root,
        apply_slots=args.dynamic_agva_slots,
        displacement_ratio_range=(args.dynamic_agva_min_displacement,
                                  args.dynamic_agva_max_displacement),
        strict_masks=args.dynamic_agva_strict_masks,
        target=getattr(args, 'dynamic_agva_target', 'fixed'),
        need_scale=getattr(args, 'dynamic_agva_need_scale', 0.20),
        ground_ratios=getattr(args, '_dynamic_agva_ground_ratios', None),
    )
    args._dynamic_agva_flag_process = process
    args._dynamic_agva_flag_process_key = cache_key
    return process


def get_train_loader_ir_flag(args, dataset, height, width, batch_size,
                             workers, num_instances, iters, trainset=None,
                             no_cam=False, train_transformer=None,
                             resolution_process=None):
    """Flag-bearing counterpart of baseline ``get_train_loader_ir``."""
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler_class = (RandomMultipleGallerySamplerNoCam if no_cam else
                         RandomMultipleGallerySampler)
        sampler = sampler_class(train_set, num_instances)
    else:
        sampler = None
    preprocessor = PreprocessorAugFlag(
        train_set, root=dataset.images_dir, transform=train_transformer,
        image_process=get_dynamic_agva_process_with_flag(
            args, dataset.images_dir),
        resolution_process=resolution_process)
    return IterLoader(DataLoader(
        preprocessor, batch_size=batch_size, num_workers=workers,
        sampler=sampler, shuffle=not rmgs_flag, pin_memory=True,
        drop_last=True), length=iters)


def _sample_path(sample, image_root):
    path = sample[0] if isinstance(sample, (tuple, list)) else sample
    return (path if osp.isabs(path) or not image_root
            else osp.join(image_root, path))


def collect_ground_geometry_ratios(samples, image_root, mask_dir,
                                   printer=print):
    loader = SCHPMaskLoader(
        mask_dir=mask_dir, image_root=image_root, resize_mismatch=True,
        strict=False)
    ratios = []
    for sample in samples:
        path = _sample_path(sample, image_root)
        mask_path = loader._find(Path(path))
        if mask_path is None:
            continue
        with Image.open(path) as image:
            size = image.size
        mask = loader.load(path, img_shape=(size[1], size[0]))
        ratio = mask_geometry_ratio(mask, size)
        if ratio is not None:
            ratios.append(ratio)
    total = len(samples)
    coverage = len(ratios) / float(total) if total else 0.0
    printer("[D-AGVA-GM] ground mask geometry coverage: {}/{} ({:.2%})"
            .format(len(ratios), total, coverage))
    if coverage < 0.90:
        raise RuntimeError(
            "ground-mixture requires at least 90% valid ground masks; "
            "got {}/{} ({:.2%})".format(len(ratios), total, coverage))
    return np.asarray(ratios, dtype=np.float64)


def _collect_aerial_ratios(samples, image_root, mask_dir):
    loader = SCHPMaskLoader(
        mask_dir=mask_dir, image_root=image_root, resize_mismatch=True,
        strict=False)
    ratios = []
    for sample in samples:
        path = _sample_path(sample, image_root)
        if loader._find(Path(path)) is None:
            continue
        with Image.open(path) as image:
            size = image.size
        ratio = mask_geometry_ratio(
            loader.load(path, img_shape=(size[1], size[0])), size)
        if ratio is not None:
            ratios.append(ratio)
    return np.asarray(ratios, dtype=np.float64)


def prepare_ground_mixture(args, aerial_samples, ground_samples,
                           aerial_root, ground_root, printer=print):
    """Build ground references and print the requested 2,000-draw preview."""
    if getattr(args, 'dynamic_agva_target', 'fixed') != 'ground-mixture':
        return None
    ground_mask_dir = getattr(args, 'dynamic_agva_ground_mask_dir', '')
    if not ground_mask_dir:
        if getattr(args, 'dataset_key', 'agreid') == 'agreidv2':
            ground_mask_dir = args.dynamic_agva_mask_dir
            args.dynamic_agva_ground_mask_dir = ground_mask_dir
        else:
            raise ValueError(
                "--dynamic-agva-ground-mask-dir is required for "
                "ground-mixture on AG-ReID.v1/LAGPeR")
    configured_image_root = getattr(args, 'dynamic_agva_image_root', '')
    effective_aerial_root = configured_image_root or aerial_root
    effective_ground_root = ground_root
    if getattr(args, 'dataset_key', 'agreid') == 'agreidv2':
        effective_ground_root = configured_image_root or ground_root
    ground_ratios = collect_ground_geometry_ratios(
        ground_samples, effective_ground_root, ground_mask_dir,
        printer=printer)
    aerial_ratios = _collect_aerial_ratios(
        aerial_samples, effective_aerial_root, args.dynamic_agva_mask_dir)
    if aerial_ratios.size == 0:
        raise RuntimeError("ground-mixture found no valid aerial mask geometry")
    rng = random.Random(getattr(args, 'seed', 1))
    scales = []
    for _ in range(2000):
        own = float(rng.choice(aerial_ratios.tolist()))
        reference = float(rng.choice(ground_ratios.tolist()))
        scales.append(adaptive_strength_scale(
            own, reference, args.dynamic_agva_need_scale))
    scales = np.asarray(scales, dtype=np.float64)
    quartiles = np.percentile(scales, [25, 50, 75])
    printer(
        "[D-AGVA-GM] 2000-draw preview: scale=0 {:.2%}, mean {:.4f}, "
        "q25/q50/q75 {:.4f}/{:.4f}/{:.4f}, ground median {:.4f}"
        .format(float(np.mean(scales <= 0.01)), float(scales.mean()),
                quartiles[0], quartiles[1], quartiles[2],
                float(np.median(ground_ratios))))
    args._dynamic_agva_ground_ratios = ground_ratios
    args._dynamic_agva_flag_process = None
    args._dynamic_agva_flag_process_key = None
    return ground_ratios


def add_agva_flag_arguments(parser):
    parser.add_argument(
        '--dynamic-agva-no-memory-update', action='store_true',
        help='let augmented aerial samples contribute loss but not memory updates')
    parser.add_argument(
        '--dynamic-agva-target', choices=('fixed', 'ground-mixture'),
        default='fixed', help='fixed legacy strength or sampled ground geometry')
    parser.add_argument(
        '--dynamic-agva-need-scale', type=float, default=0.20,
        help='geometry gap corresponding to full ground-mixture strength')
    parser.add_argument(
        '--dynamic-agva-ground-mask-dir', default='',
        help='ground SCHP masks; v2 defaults to the aerial/all mask root')


def validate_agva_flag_args(args):
    active = agva_flag_loader_enabled(args)
    if active and not getattr(args, 'use_dynamic_agva', False):
        raise ValueError(
            'D-AGVA memory/target switches require --use-dynamic-agva')
    if float(getattr(args, 'dynamic_agva_need_scale', 0.20)) <= 0:
        raise ValueError('--dynamic-agva-need-scale must be positive')
